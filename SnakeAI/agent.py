# agent.py
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 1_000_000
BATCH_SIZE = 10000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

        # Define grid size based on game dimensions (assuming game is initialized later)
        self.grid_height_cells = 24  # Example for 480 height / 20 block size, adjust if needed
        self.grid_width_cells = 32   # Example for 640 width / 20 block size, adjust if needed
        input_size_features = 11 # Number of features in feature-based state
        input_size_grid = self.grid_height_cells * self.grid_width_cells
        combined_input_size = input_size_features + input_size_grid # Calculate combined input size


        self.model = Linear_QNet(combined_input_size, 256, 3) # Input size now combined
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        feature_state = self.get_state_features(game) # Get feature-based state
        grid_state = self.get_state_grid(game) # Get grid-based state
        combined_state = np.concatenate((feature_state, grid_state)) # Concatenate them
        return combined_state


    def get_state_features(self, game): # Feature-based state function
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        return np.array(state, dtype=int)


    def get_state_grid(self, game): # Grid-based state function
        grid_size = (game.h // game.block_size, game.w // game.block_size)
        game_grid = np.zeros(grid_size, dtype=int)

        food_x_grid = game.food.x // game.block_size
        food_y_grid = game.food.y // game.block_size
        # Make sure food indices are within grid bounds
        if 0 <= food_y_grid < grid_size[0] and 0 <= food_x_grid < grid_size[1]:
            game_grid[food_y_grid, food_x_grid] = 3

        for i, point in enumerate(game.snake):
            snake_x_grid = point.x // game.block_size
            snake_y_grid = point.y // game.block_size

            #print(f"Inside get_state_grid - Before int cast: snake_x_grid={snake_x_grid}, snake_y_grid={snake_y_grid}, types={type(snake_x_grid)}, {type(snake_y_grid)}") # Debug print BEFORE cast

            snake_x_grid = int(snake_x_grid)
            snake_y_grid = int(snake_y_grid)

            #print(f"Inside get_state_grid - After int cast: snake_x_grid={snake_x_grid}, snake_y_grid={snake_y_grid}, types={type(snake_x_grid)}, {type(snake_y_grid)}") # Debug print AFTER cast
            #print(f"Inside get_state_grid - Game grid shape: {game_grid.shape}") # Debug print grid shape

            # Make sure snake indices are within grid bounds, and clamp them if needed
            snake_y_grid = max(0, min(snake_y_grid, grid_size[0] - 1))
            snake_x_grid = max(0, min(snake_x_grid, grid_size[1] - 1))

            #print(f"Inside get_state_grid - After clamping: snake_x_grid={snake_x_grid}, snake_y_grid={snake_y_grid}") # Debug print AFTER clamp

            if i == 0:
                game_grid[snake_y_grid, snake_x_grid] = 2 # Line where error occurs
            else:
                game_grid[snake_y_grid, snake_x_grid] = 1

        return game_grid.flatten()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI() # Initialize the game here, after Agent is created
    agent.grid_height_cells = game.h // game.block_size # Correctly set grid dimensions based on game
    agent.grid_width_cells = game.w // game.block_size

    print("Starting training. Grid size:", (agent.grid_height_cells, agent.grid_width_cells),
          "Feature input size: 11, Grid input size:", agent.grid_height_cells * agent.grid_width_cells,
          "Combined input size to model:", 11 + agent.grid_height_cells * agent.grid_width_cells)


    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move, agent.n_games % 20 == 0) # Removed frame_iteration argument
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()