import argparse
from agents.monte_carlo_agent import TicTacToeMonteCarlo
from agents.q_learning_agent import TicTacToeQlearning
from agents.dqn_agent import DQNAgent

AGENTS = {'mc': TicTacToeMonteCarlo, 'q': TicTacToeQlearning, 'dqn': DQNAgent}

def evaluate(agent, games):
    w=t=l=0
    for _ in range(games):
        r = agent.gameplay()
        if r==1: w+=1
        elif r==0: t+=1
        else: l+=1
    return w,t,l

def test_hyper(agent_key, param, values, episodes, games):
    for v in values:
        kwargs={param: float(v)}
        if agent_key=='dqn':
            kwargs.update({'state_size':9,'action_size':9})
        agent = AGENTS[agent_key](**kwargs)
        agent.train(episodes)
        w,t,l = evaluate(agent, games)
        print(f"{agent_key} {param}={v}: Win={w/games*100:.2f}%, Tie={t/games*100:.2f}%, Loss={l/games*100:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent','-a',choices=['mc','q','dqn','all'],required=True)
    parser.add_argument('--mode','-m',choices=['play','test_hyper_param'],default='play')
    parser.add_argument('--model','-p',type=str,default=None)
    parser.add_argument('--episodes','-e',type=int,default=100000)
    parser.add_argument('--games','-g',type=int,default=1000)
    parser.add_argument('--param-name',type=str,default=None)
    parser.add_argument('--param-values',type=str,default=None)
    args = parser.parse_args()
    if args.mode=='play':
        if args.agent!='all':
            path = args.model or f"models/{args.agent}_model.pkl"
            agent = AGENTS[args.agent]
            if args.agent == 'dqn':
                agent = agent.load(path)
            else:
                agent.load(path)
            w,t,l = evaluate(agent, args.games)
            print(f"Win={w/args.games*100:.2f}%, Tie={t/args.games*100:.2f}%, Loss={l/args.games*100:.2f}%")
    else:
        vals = args.param_values.split(',')
        ags = ['mc','q','dqn'] if args.agent=='all' else [args.agent]
        for ak in ags:
            test_hyper(ak, args.param_name, vals, args.episodes, args.games)

if __name__=='__main__':
    main()
