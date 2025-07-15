import numpy as np
import random
from env import TicTacToe
from config import CONFIG

class TicTacToeMonteCarlo:
    def __init__(self, size=3, learning_rate=CONFIG['alpha_mc'], discount_factor=CONFIG['discount_factor_mc'], epsilon=CONFIG['epsilon_mc']):
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
        self.val_sample_every = CONFIG['val_sample_every_mc']

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

    def update_q_table(self, episode, reward):
        visited_states = set()
        for state, action in episode:
            state_str = self.state_to_str(state)
            if state_str not in visited_states:
                self.q_table.setdefault(state_str, {})
                self.q_table[state_str].setdefault(action, 0)
                self.q_table[state_str][action] += self.learning_rate * (reward - self.q_table[state_str][action])
                visited_states.add(state_str)

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
            valid_move = self.game.make_move(square, letter)
            if not valid_move:
                return -1
            if self.game.current_winner:
                return 1 if letter == self.player_1 else -1
            current_state = self.game.board.copy()
            letter = self.player_2 if letter == self.player_1 else self.player_1
        return 0

    def train(self, episodes):
        for ep in range(episodes):
            self.game = TicTacToe(self.size)
            current_state = self.game.board.copy()
            episode = []
            total_reward = 0
            letter = self.player_1
            while self.game.empty_squares():
                moves = self.game.available_moves()
                if letter == self.player_2:
                    square = random.choice(moves)
                else:
                    square = self.select_action(current_state, moves)
                    if self.game.num_empty_squares() == self.size**2:
                        square = random.choice(moves)
                episode.append((current_state, square))
                valid_move = self.game.make_move(square, letter)
                if not valid_move:
                    break
                if self.game.current_winner:
                    reward = CONFIG['win_reward'] if letter == self.player_1 else CONFIG['lose_reward']
                    break
                else:
                    reward = CONFIG['step_reward']
                total_reward += reward
                current_state = self.game.board.copy()
                letter = self.player_2 if letter == self.player_1 else self.player_1
            else:
                reward = CONFIG['tie_reward']
            total_reward += reward
            self.update_q_table(episode, total_reward)
            self.reward_per_episode.append(total_reward)
            if ep % self.val_sample_every == 0:
                self._validate(ep)

    def _validate(self, ep):
        results = [str(self.gameplay()) for _ in range(10000)]
        wins = results.count('1')
        ties = results.count('0')
        loses = results.count('-1')
        score = wins - loses
        if score > self.best_score:
            self.best_q_table = dict(self.q_table)
            self.best_score = score
            self.best_episode = ep
        self.validation.append((ep, (wins, ties, loses)))

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)