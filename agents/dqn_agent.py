import numpy as np
import random
from collections import deque
from tqdm import tqdm
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense, InputLayer # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from env import TicTacToe
from config import CONFIG

# Reward structure for game outcomes and steps
REWARDS = CONFIG.get('rewards', {'win': 1, 'lose': -10, 'tie': 0.5, 'step': 0.002})

class DQNAgent:
    def __init__(
        self,
        state_size=9,
        action_size=9,
        learning_rate=None,
        gamma=None,
        epsilon=None,
        epsilon_decay=None,
        epsilon_min=None,
        batch_size=None,
        buffer_size=None,
        algorithm=None
    ):
        # Hyperparameters from config or provided overrides
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate if learning_rate is not None else CONFIG['lr_dqn']
        self.gamma = gamma if gamma is not None else CONFIG['gamma_dqn']
        self.epsilon = epsilon if epsilon is not None else CONFIG['epsilon_dqn']
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else CONFIG.get('epsilon_decay_dqn', CONFIG.get('epsilon_decay_q', 0.995))
        self.epsilon_min = epsilon_min if epsilon_min is not None else CONFIG.get('epsilon_min_dqn', CONFIG.get('epsilon_min_q', 0.01))
        self.batch_size = batch_size if batch_size is not None else CONFIG['batch_size']
        self.memory = deque(maxlen=buffer_size if buffer_size is not None else CONFIG['buffer_size'])
        self.algorithm = (algorithm or 'DQN').upper()  # 'DQN' or 'DDQN'

        # Build the Q-network and the target network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential([
            InputLayer(shape=(self.state_size,)),
            Dense(CONFIG['hidden1'], activation=CONFIG['activation']),
            Dense(CONFIG['hidden2'], activation=CONFIG['activation']),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(self.learning_rate), loss='mse')
        return model

    @staticmethod
    def reshape_state(board):
        arr = [0 if c==' ' else 1 if c=='X' else -1 for c in board]
        return np.array(arr).reshape(1, -1)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, moves):
        if np.random.rand() < self.epsilon:
            return random.choice(moves)
        q_vals = self.model.predict(state, verbose=0)[0]
        valid = {m: q_vals[m] for m in moves}
        return max(valid, key=valid.get)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        X, y = [], []
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                if self.algorithm == 'DDQN':
                    next_act = np.argmax(self.model.predict(next_state, verbose=0)[0])
                    next_q = self.target_model.predict(next_state, verbose=0)[0][next_act]
                    target += self.gamma * next_q
                else:
                    target += self.gamma * np.max(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)[0]
            target_f[action] = target
            X.append(state.flatten())
            y.append(target_f)
        self.model.fit(np.array(X), np.array(y), epochs=1, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, episodes=None):
        episodes = episodes if episodes is not None else CONFIG['episodes']
        size = int(np.sqrt(self.state_size))
        for ep in tqdm(range(episodes), desc=f"Training {self.algorithm}"):
            game = TicTacToe(size)
            state = game.board.copy()
            letter = 'X'
            last_state = None
            last_action = None
            done = False
            while game.empty_squares():
                st = self.reshape_state(state)
                moves = game.available_moves()
                action = random.choice(moves) if letter=='O' else self.act(st, moves)
                next_board = game.make_move(action, letter)
                next_board = game.board.copy()
                nxt = self.reshape_state(next_board)

                if game.current_winner:
                    reward = REWARDS['win'] if letter=='X' else REWARDS['lose']
                    done = True
                else:
                    reward = REWARDS['step']
                    last_state, last_action = st, action

                self.remember(st, action, reward, nxt, done)
                self.replay()
                state = next_board.copy()
                letter = 'O' if letter=='X' else 'X'
                if done:
                    break

            if not game.current_winner:
                reward = REWARDS['tie']
                self.remember(last_state, last_action, reward, nxt, True)
                self.replay()

            if ep % CONFIG['update_freq'] == 0:
                self.update_target_model()

    def gameplay(self):
        orig = self.epsilon
        self.epsilon = 0.0
        size = int(np.sqrt(self.state_size))
        game = TicTacToe(size)
        letter = 'X'
        state = game.board.copy()
        while game.empty_squares():
            st = self.reshape_state(state)
            moves = game.available_moves()
            action = random.choice(moves) if letter=='O' else self.act(st, moves)
            game.make_move(action, letter)
            state = game.board.copy()
            if game.current_winner:
                self.epsilon = orig
                return 1 if letter=='X' else -1
            letter = 'O' if letter=='X' else 'X'
        self.epsilon = orig
        return 0

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, path, algorithm=None):
        agent = cls(
            state_size= 9,
            action_size=9,
            learning_rate=CONFIG['lr_dqn'],
            gamma=CONFIG['gamma_dqn'],
            epsilon=CONFIG['epsilon_dqn'],
            epsilon_decay=CONFIG.get('epsilon_decay_dqn', CONFIG.get('epsilon_decay_q', 0.995)),
            epsilon_min=CONFIG.get('epsilon_min_dqn', CONFIG.get('epsilon_min_q', 0.01)),
            batch_size=CONFIG['batch_size'],
            buffer_size=CONFIG['buffer_size'],
            algorithm=algorithm
        )
        agent.model = load_model(path)
        agent.target_model = load_model(path)
        return agent