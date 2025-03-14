import cProfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import os

# --- Hyperparameters ---
GAMMA = 0.99
LR_ACTOR = 0.0001  # Adjusted learning rate
LR_CRITIC = 0.0002 # Adjusted learning rate
HIDDEN_SIZE = 256  # Not used anymore, but kept for clarity
BATCH_SIZE = 512    # Adjusted batch size
UPDATE_ITERATIONS = 10
CLIP_PARAM = 0.2
MAX_GRAD_NORM = 0.5
SEED = 42

# --- Seed Setting ---
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Device Handling ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Actor ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)

        # Weight and bias initialization
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.normal_(self.fc3.weight, std=0.01)
        nn.init.normal_(self.mean.weight, std=0.01)
        nn.init.normal_(self.log_std.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.bias)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        pre_tanh_action = normal.sample()
        action = torch.tanh(pre_tanh_action)
        log_prob = normal.log_prob(pre_tanh_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

# --- Critic ---
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.value = nn.Linear(64, 1)

        # Weight and bias initialization
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.normal_(self.fc3.weight, std=0.01)
        nn.init.normal_(self.value.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value(x)
        return value

# --- PPO Agent ---
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.memory = []

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def get_action(self, state):
        return self.actor.get_action(state)

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch, old_log_prob_batch = zip(*self.memory)

        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32).to(device)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32).to(device)
        reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).unsqueeze(1).to(device)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32).to(device)
        done_batch = torch.tensor(np.array(done_batch), dtype=torch.float32).unsqueeze(1).to(device)
        old_log_prob_batch = torch.tensor(np.array(old_log_prob_batch), dtype=torch.float32).unsqueeze(-1).to(device)

        # Calculate GAE
        with torch.no_grad():
            values = self.critic(state_batch)
            next_values = self.critic(next_state_batch)
            deltas = reward_batch + GAMMA * (1 - done_batch) * next_values - values
            advantages = torch.zeros_like(deltas)
            advantage = 0
            for t in reversed(range(len(deltas))):
                advantage = deltas[t] + GAMMA * 0.90 * (1 - done_batch[t]) * advantage  # GAE with lambda=0.90
                advantages[t] = advantage
            target_v = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(UPDATE_ITERATIONS):
            mean, log_std = self.actor(state_batch)
            std = log_std.exp()
            normal = Normal(mean, std)
            new_log_prob = normal.log_prob(action_batch).sum(dim=1, keepdim=True)

            ratio = torch.exp(new_log_prob - old_log_prob_batch)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(target_v, self.critic(state_batch))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.critic_optimizer.step()

        self.memory = []

# --- Training ---
def train():
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array") # Increased max steps
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim)
    n_episodes = 1000
    render_interval = 100

    # --- Load Models (Corrected) ---
    actor_path = 'actor.pth'
    critic_path = 'critic.pth'

    # Check if the files exist before loading
    if os.path.exists(actor_path):
        agent.actor.load_state_dict(torch.load(actor_path))
        print("Loaded actor model from", actor_path)
    else:
        print("Actor model file not found. Starting from scratch.")

    if os.path.exists(critic_path):
        agent.critic.load_state_dict(torch.load(critic_path))
        print("Loaded critic model from", critic_path)
    else:
        print("Critic model file not found.  Starting from scratch.")

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action, log_prob = agent.get_action(state_tensor)
            action_np = action.detach().cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            agent.store_transition(state, action_np, reward, next_state, done, log_prob.detach().cpu().numpy()[0])
            state = next_state
            total_reward += reward


            if episode % render_interval == 0:
                img = env.render()
                plt.imshow(img)
                plt.title(f"Episode: {episode}")
                plt.pause(0.01)
                plt.clf()

        agent.update()
        print(f"Episode: {episode}, Total Reward: {total_reward}")

    env.close()

    # --- Save Models (Corrected) ---
    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)
    print(f"Models saved to {actor_path} and {critic_path}")

if __name__ == "__main__":
    train()