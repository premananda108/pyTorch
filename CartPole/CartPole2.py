import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        # Инициализация весов
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        return self.network(x)

def train_episode(env, policy, optimizer, episode):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    values = []
    
    # Уменьшаем случайность с течением времени
    exploration_noise = max(0.3 * (1 - episode / 500), 0.01)
    
    for t in range(200):
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs = policy(state_tensor)
            
            # Добавляем исследование
            if np.random.random() < exploration_noise:
                action = env.action_space.sample()
            else:
                dist = Categorical(action_probs)
                action = dist.sample().item()
        
        # Повторно вычисляем вероятности для градиента
        action_probs = policy(state_tensor)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(torch.tensor(action))
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Модифицируем награду для лучшего обучения
        x, x_dot, theta, theta_dot = next_state
        x_threshold = env.unwrapped.x_threshold
        theta_threshold = env.unwrapped.theta_threshold_radians
        
        # Даем больше награды за центральное положение
        position_reward = 1.0 - abs(x) / x_threshold
        angle_reward = 1.0 - abs(theta) / theta_threshold
        
        modified_reward = reward * (0.8 + 0.1 * position_reward + 0.1 * angle_reward)
        
        log_probs.append(log_prob)
        rewards.append(modified_reward)
        
        state = next_state
        
        if terminated or truncated:
            break
    
    # Расчет возвратов с нормализацией
    returns = []
    R = 0
    gamma = 0.99
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    # Расчет функции потерь с клиппированием
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    
    if len(policy_loss) > 0:
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        
        # Клиппирование градиентов
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        
        loss.backward()
        optimizer.step()
    
    return sum(rewards), len(rewards)

def plot_rewards(rewards, window_size=100):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.6, label='Награды')
    plt.plot(np.convolve(rewards, np.ones(window_size)/window_size, mode='valid'), 
             label='Скользящее среднее')
    plt.title('Награды за эпизоды')
    plt.xlabel('Эпизод')
    plt.ylabel('Награда')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('rewards_plot.png')
    plt.close()

# Настройка окружения
env = gym.make('CartPole-v1', render_mode="rgb_array")
env = RecordVideo(env, "video", episode_trigger=lambda x: x % 100 == 0)

# Инициализация политики и оптимизатора
input_dim = 4
output_dim = 2
policy = Policy(input_dim, output_dim)
optimizer = optim.Adam(policy.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

# Обучение
n_episodes = 1000
total_rewards = []
episode_lengths = []
best_reward = 0
best_policy_state = None
eval_interval = 10

# Добавляем экспоненциальное скользящее среднее для отслеживания прогресса
ema_reward = None
alpha = 0.1

for episode in range(n_episodes):
    reward, length = train_episode(env, policy, optimizer, episode)
    total_rewards.append(reward)
    episode_lengths.append(length)
    
    # Обновляем экспоненциальное среднее
    if ema_reward is None:
        ema_reward = reward
    else:
        ema_reward = alpha * reward + (1 - alpha) * ema_reward
    
    # Сохраняем модель только если она существенно лучше
    if reward > best_reward and length >= 195 and episode > 100:
        if best_reward == 0 or reward > best_reward * 1.1:  # Минимум 10% улучшение
            best_reward = reward
            best_policy_state = policy.state_dict().copy()
    
    if episode % eval_interval == 0:
        avg_reward = np.mean(total_rewards[-eval_interval:])
        avg_length = np.mean(episode_lengths[-eval_interval:])
        print(f'Эпизод {episode}, Текущая награда: {reward:.2f}, EMA: {ema_reward:.2f}, Средняя длина: {avg_length:.1f}, Лучшая награда: {best_reward:.2f}')

print(f'Среднее полное вознаграждение в {n_episodes} эпизодах: {sum(total_rewards) / n_episodes:.2f}')

if best_policy_state is not None:
    policy.load_state_dict(best_policy_state)
    torch.save(best_policy_state, 'best_policy.pth')

plot_rewards(total_rewards)
env.close()