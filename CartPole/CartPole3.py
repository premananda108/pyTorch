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

def calculate_stability_score(states):
    # Вычисляем среднее отклонение позиции и угла
    x_positions = np.array([state[0] for state in states])
    thetas = np.array([state[2] for state in states])
    x_velocities = np.array([state[1] for state in states])
    theta_velocities = np.array([state[3] for state in states])
    
    position_stability = 1.0 / (1.0 + np.std(x_positions))
    angle_stability = 1.0 / (1.0 + np.std(thetas))
    velocity_stability = 1.0 / (1.0 + np.std(x_velocities) + np.std(theta_velocities))
    
    return (position_stability + angle_stability + velocity_stability) / 3.0

def train_episode(env, policy, optimizer, episode):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    states = []
    
    # Уменьшаем исследование, так как у нас уже есть хорошая базовая политика
    exploration_noise = max(0.1 * (1 - episode / 200), 0.01)
    
    for t in range(200):
        states.append(state)
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            action_probs = policy(state_tensor)
            if np.random.random() < exploration_noise:
                action = env.action_space.sample()
            else:
                dist = Categorical(action_probs)
                action = dist.sample().item()
        
        # Создаем новый тензор для градиентов
        action_probs = policy(state_tensor)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(torch.tensor(action, dtype=torch.int64))
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Модифицируем награду для поощрения стабильности
        x, x_dot, theta, theta_dot = next_state
        x_threshold = env.unwrapped.x_threshold
        theta_threshold = env.unwrapped.theta_threshold_radians
        
        # Штраф за отклонение от центра и движение
        position_penalty = (abs(x) / x_threshold) ** 2
        angle_penalty = (abs(theta) / theta_threshold) ** 2
        velocity_penalty = 0.1 * (abs(x_dot) + abs(theta_dot))
        
        # Награда за стабильность
        stability_reward = 1.0 - 0.4 * position_penalty - 0.4 * angle_penalty - 0.2 * velocity_penalty
        
        # Комбинируем с базовой наградой
        modified_reward = reward * (0.5 + 0.5 * stability_reward)
        
        log_probs.append(log_prob)
        rewards.append(modified_reward)
        state = next_state
        
        if terminated or truncated:
            break
    
    # Добавляем бонус за стабильность в конце эпизода
    stability_score = calculate_stability_score(states)
    if len(states) >= 195:  # Если эпизод был достаточно длинным
        rewards.append(stability_score * 10.0)  # Бонус за общую стабильность
    
    # Вычисляем возвраты
    returns = []
    R = 0
    gamma = 0.99
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    # Обрезаем returns до длины log_probs
    returns = returns[:len(log_probs)]
    
    # Вычисляем loss
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    
    if policy_loss:
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
    
    return sum(rewards[:-1]), len(states), stability_score

def plot_rewards_and_stability(rewards, stability_scores, window_size=100):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # График наград
    ax1.plot(rewards, alpha=0.6, label='Награды')
    ax1.plot(np.convolve(rewards, np.ones(window_size)/window_size, mode='valid'), 
             label='Скользящее среднее')
    ax1.set_title('Награды за эпизоды')
    ax1.set_xlabel('Эпизод')
    ax1.set_ylabel('Награда')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График стабильности
    ax2.plot(stability_scores, alpha=0.6, label='Стабильность')
    ax2.plot(np.convolve(stability_scores, np.ones(window_size)/window_size, mode='valid'),
             label='Скользящее среднее')
    ax2.set_title('Стабильность')
    ax2.set_xlabel('Эпизод')
    ax2.set_ylabel('Оценка стабильности')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stability_plot.png')
    plt.close()

env = gym.make('CartPole-v1', render_mode="rgb_array")
env = RecordVideo(env, "video", episode_trigger=lambda x: x % 100 == 0)

input_dim = 4
output_dim = 2
policy = Policy(input_dim, output_dim)
# Загружаем предварительно обученную модель
policy.load_state_dict(torch.load('best_policy.pth'))

# Используем меньшую скорость обучения для тонкой настройки
optimizer = optim.Adam(policy.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)

# Обучение
n_episodes = 500
total_rewards = []
stability_scores = []
episode_lengths = []
best_reward = 0
best_stability = 0
best_policy_state = None
eval_interval = 10

for episode in range(n_episodes):
    reward, length, stability = train_episode(env, policy, optimizer, episode)
    total_rewards.append(reward)
    stability_scores.append(stability)
    episode_lengths.append(length)
    
    # Сохраняем модель, если она показывает хорошую стабильность
    if length >= 195 and stability > best_stability:
        best_stability = stability
        best_policy_state = policy.state_dict().copy()
    
    if episode % eval_interval == 0:
        avg_reward = np.mean(total_rewards[-eval_interval:])
        avg_stability = np.mean(stability_scores[-eval_interval:])
        print(f'Эпизод {episode}, Награда: {reward:.2f}, Стабильность: {stability:.3f}, '
              f'Средняя стабильность: {avg_stability:.3f}')

print(f'Финальная средняя стабильность: {np.mean(stability_scores[-100:]):.3f}')

if best_policy_state is not None:
    policy.load_state_dict(best_policy_state)
    torch.save(best_policy_state, 'stable_policy.pth')

plot_rewards_and_stability(total_rewards, stability_scores)
env.close()