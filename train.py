import argparse, os
from agents.monte_carlo_agent import TicTacToeMonteCarlo
from agents.q_learning_agent import TicTacToeQlearning
from agents.dqn_agent import DQNAgent

AGENTS = {'mc': TicTacToeMonteCarlo, 'q': TicTacToeQlearning, 'dqn': DQNAgent}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', '-a', choices=['mc','q','dqn'], required=True)
    parser.add_argument('--episodes', '-e', type=int, default=100000)
    parser.add_argument('--save-path', '-o', type=str, default=None)
    args = parser.parse_args()

    agent = AGENTS[args.agent]()
    agent.train(args.episodes)
    os.makedirs('models', exist_ok=True)
    path = args.save_path or f"models/{args.agent}_model.pkl"
    if args.agent == 'dqn':
        path = args.save_path or f"models/{args.agent}_model.keras"
    agent.save(path)
    print(f"Saved model to {path}")

if __name__ == '__main__':
    main()
