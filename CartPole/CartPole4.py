import gymnasium as gym
import torch
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode=None)

n_state = env.observation_space.shape[0]
n_action = env.action_space.n

def run_episode(env, weight):
    state, _ = env.reset()
    total_reward = 0
    for _ in range(200):
        state = torch.from_numpy(state).float()  # Явное преобразование
        action = torch.argmax(torch.matmul(state, weight))
        state, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

total_rewards = []
best_total_reward = 0
best_weight = None
n_episode = 1000

for episode in range(n_episode):
     weight = torch.rand(n_state, n_action)
     total_reward = run_episode(env, weight)
     print('Эпизод {}: {}'.format(episode+1, total_reward))
     if total_reward > best_total_reward:
         best_weight = weight
         best_total_reward = total_reward
     total_rewards.append(total_reward)
     if best_total_reward == 200:
         break

print('Среднее полное вознаграждение в {} эпизодах: {}'.format(
n_episode, sum(total_rewards) / n_episode))

plt.plot(total_rewards)
plt.xlabel('Эпизод')
plt.show()
plt.ylabel('Вознаграждение')

n_episode_eval = 100
total_rewards_eval = []
for episode in range(n_episode_eval):
     total_reward = run_episode(env, best_weight)
     print('Эпизод {}: {}'.format(episode+1, total_reward))
     total_rewards_eval.append(total_reward)

print('Среднее полное вознаграждение в {} эпизодах: {}'.format(
n_episode_eval, sum(total_rewards_eval) / n_episode_eval))
