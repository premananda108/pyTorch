import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.recent_moves = deque(maxlen=32)  # Память для последних  ходов

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - game.block_size, head.y)
        point_r = Point(head.x + game.block_size, head.y)
        point_u = Point(head.x, head.y - game.block_size)
        point_d = Point(head.x, head.y + game.block_size)

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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        random_move_choice = random.randint(0, 200)

        random_move = random_move_choice < self.epsilon

        if random_move:
            move = random.randint(0, 2)  # Случайный ход
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()  # Лучший ход по мнению модели

        # Проверка на повторение поворотов, если это не случайный ход
        if not random_move:
            turn_threshold = 2  # Максимальное количество поворотов в одну сторону за последние n ходов
            move_type = ["straight", "right", "left"][move]  # Типы ходов для читаемости

            if move != 0:  # Если выбран поворот (вправо или влево)
                turn_direction = ["right", "left"][move - 1]  # Направление поворота
                recent_turns_in_direction = 0

                # Считаем последние повороты в выбранном направлении
                for recent_move_index in self.recent_moves:
                    recent_move_type = ["straight", "right", "left"][recent_move_index]
                    if recent_move_type == turn_direction:
                        recent_turns_in_direction += 1

                if recent_turns_in_direction >= turn_threshold:
                    # Слишком много поворотов в этом направлении, ищем альтернативу
                    possible_moves = [0, 1, 2]  # 0: прямо, 1: вправо, 2: влево
                    possible_moves.remove(move)  # Исключаем текущий "проблемный" поворот

                    best_alternative_move = -1
                    best_alternative_index = -1

                    # Сначала пробуем идти прямо
                    if 0 in possible_moves:
                        straight_safe = not state[0]  # Проверяем опасность прямо (state[0])
                        if straight_safe:
                            best_alternative_move = [1, 0, 0]  # Прямо
                            best_alternative_index = 0
                            possible_moves.remove(0)  # Исключаем из дальнейших проверок

                    if best_alternative_index == -1:  # Если прямо не безопасно или не было в possible_moves
                        # Выбираем оставшийся альтернативный поворот (если есть)
                        if possible_moves:
                            alternative_move_index = possible_moves[0]  # Берем первый оставшийся индекс
                            alternative_move_type = ["straight", "right", "left"][alternative_move_index]

                            # Определяем, какой danger index в state соответствует alternative_move_type
                            danger_index = -1

                            # Получаем информацию о направлении из state
                            dir_l, dir_r, dir_u, dir_d = state[3:7]
                            current_direction_index = [dir_l, dir_r, dir_u, dir_d].index(
                                1)  # 1 соответствует True, так как state содержит int

                            if alternative_move_type == "right":
                                danger_index = 1  # Опасность справа
                            elif alternative_move_type == "left":
                                danger_index = 2  # Опасность слева
                            elif alternative_move_type == "straight":  # Хотя прямо уже проверили выше, на всякий случай
                                danger_index = 0  # Опасность прямо

                            if danger_index != -1:
                                alternative_safe = not state[danger_index]  # Проверяем безопасность альтернативы
                                if alternative_safe:
                                    best_alternative_move_index = alternative_move_index
                                    best_alternative_move = [0, 0, 0]
                                    best_alternative_move[best_alternative_move_index] = 1
                                    best_alternative_index = best_alternative_move_index

                    if best_alternative_index != -1:
                        final_move = best_alternative_move
                        move = best_alternative_index
                    else:
                        final_move[
                            move] = 1  # Если нет безопасных альтернатив, используем исходный (возможно, опасный) ход
                else:
                    final_move[
                        move] = 1  # Используем изначально выбранный поворот, если нет превышения лимита поворотов
            else:  # Если изначально был выбран ход прямо, просто используем его
                final_move[move] = 1
        else:  # Если ход был выбран случайно (epsilon-exploration), просто используем его
            final_move[move] = 1

        self.recent_moves.append(move)

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move, agent.n_games % 20 == 0)
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