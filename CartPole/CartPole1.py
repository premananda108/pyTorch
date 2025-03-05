import gymnasium as gym
#from gymnasium.wrappers import RecordVideo

env = gym.make('CartPole-v1')
#env = gym.make('CartPole-v1', render_mode="human")
#env = gym.make('CartPole-v1', render_mode="rgb_array")
#env = RecordVideo(env, "video")

total_rewards = []
n_episode = 10000

for episod in range(n_episode):
    state = env.reset()
    total_reward = 0
    for _ in range(200):
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            observation, info = env.reset()
            break
    total_rewards.append(total_reward)
    if episod % 100 == 0:
        print('Эпизод: '+str(episod))
print('Среднее полное вознаграждение в {} эпизодах: {}'.format(
n_episode, sum(total_rewards) / n_episode))