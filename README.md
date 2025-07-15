# Tic-Tac-Toe Reinforcement Learning Agents

This repository contains implementations of three RL agents for Tic-Tac-Toe:
- Monte Carlo
- Q-Learning
- Deep Q-Network (DQN) (with optional Double DQN)

## Setup

```bash
pip install numpy tqdm tensorflow pytest jupyter
```

## Structure

```
tic_tac_toe_rl/
├── README.md
├── requirements.txt
├── config.py
├── env.py
├── train.py
├── evaluate.py
├── agents/
│   ├── __init__.py
│   ├── monte_carlo_agent.py
│   ├── q_learning_agent.py
│   └── dqn_agent.py
├── tests/
│   ├── test_env.py
│   └── test_agents.py
└── notebooks/
    └── Project_4_TicTacToe-Ianivkk.ipynb
```

## Training

```bash
python train.py --agent mc --episodes 100000
python train.py --agent q --episodes 200000
python train.py --agent dqn --episodes 50000
```

## Evaluation

Play mode:
```bash
python evaluate.py --agent mc --mode play --games 1000 --model models/mc_model.pkl
```

Hyperparameter tests:
```bash
python evaluate.py --agent all --mode test_hyper_param \
  --param-name epsilon --param-values 0.1,0.35,0.7,1.0 \
  --episodes 50000 --games 1000
```

