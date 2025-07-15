import numpy as np
import random
from tqdm import tqdm
from env import TicTacToe
from config import CONFIG

class TicTacToeQlearning:
    def __init__(self, size=3, learning_rate=CONFIG['alpha_q'], discount_factor=CONFIG['discount_factor_q'], epsilon=CONFIG['epsilon_q']):
        self.size = size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.best_q_table = {}
        self.game = TicTacToe(size)
        self.player_1 = 'X'
        self.player_2 = 'O'
        self.reward_per_episode = []
        self.best_score = 0
        self.best_episode = 0
        self.validation = []
        self.val_sample_every = CONFIG['val_sample_every_q']

    def state_to_str(self, state):
        return ''.join(state)

    def select_action(self, state, moves):
        state_str = self.state_to_str(state)
        if np.random.rand() < self.epsilon:
            return random.choice(moves)
        if state_str not in self.q_table:
            return random.choice(moves)
        filtered_items = [(key, value) for key, value in self.q_table[state_str].items() if key in moves]
        return max(filtered_items, key=lambda item: item[1])[0]

    def update_q_table(self, state, action, reward, next_state, moves):
        state_str = self.state_to_str(state)
        next_state_str = self.state_to_str(next_state)

        if state_str not in self.q_table:
            self.q_table[state_str] = {m: 0 for m in moves}

        next_moves = self.game.available_moves() or self.game.total_moves()
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = {m: 0 for m in next_moves}

        max_next = max(self.q_table[next_state_str].values(), default=0)
        self.q_table[state_str][action] += self.learning_rate * (reward + self.discount_factor * max_next - self.q_table[state_str][action])

    def gameplay(self):
        self.game = TicTacToe(self.size)
        current_state = self.game.board.copy()
        letter = self.player_1
        while self.game.empty_squares():
            moves = self.game.available_moves()
            if letter == self.player_2:
                square = random.choice(moves)
            else:
                square = self.select_action(current_state, moves)
            valid = self.game.make_move(square, letter)
            if not valid:
                return -1
            if self.game.current_winner:
                return 1 if letter == self.player_1 else -1
            current_state = self.game.board.copy()
            letter = self.player_2 if letter == self.player_1 else self.player_1
        return 0

    def train(self, episodes):
        for e in tqdm(range(episodes)):
            self.game = TicTacToe(self.size)
            current_state = self.game.board.copy()
            total_reward = 0
            letter = self.player_1
            while self.game.empty_squares():
                moves = self.game.available_moves()
                if letter == self.player_2:
                    square = random.choice(moves)
                else:
                    square = self.select_action(current_state, moves)
                next_state = self.game.make_move(square, letter)
                next_state = self.game.board.copy()
                if self.game.current_winner:
                    reward = CONFIG['win_reward']
                    break
                elif not self.game.empty_squares():
                    reward = CONFIG['tie_reward']
                else:
                    reward = CONFIG['step_reward']
                self.update_q_table(current_state, square, reward, next_state, moves)
                total_reward += reward
                current_state = self.game.board.copy()
                letter = self.player_2 if letter == self.player_1 else self.player_1
            self.epsilon = max(CONFIG['epsilon_min_q'], self.epsilon * CONFIG['epsilon_decay_q'])
            self.reward_per_episode.append(total_reward)
            if e % self.val_sample_every == 0:
                self._validate(e)

    def _validate(self, e):
        results = [str(self.gameplay()) for _ in range(10000)]
        wins = results.count('1')
        ties = results.count('0')
        loses = results.count('-1')
        score = wins - loses
        if score > self.best_score:
            self.best_score = score
            self.best_episode = e
        self.validation.append((e, (wins, ties, loses)))

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)