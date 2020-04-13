from Agents.dqn_agent import DQN_Agent, DQN_C51Agent
from Wrapper import wrapper_new
import numpy as np
import torch
import gym
import argparse
import time


def play_atari(eps):
    for ep in range(eps):
        score = 0
        state = env.reset()
        while True:
            env.render()
            action = agent.act(state, 0.)
            state, reward, done, _ = env.step(action) 
            score += reward
            time.sleep(0.05)
            if done:
                break
        print("Episode {} | Score: {}".format(ep, score))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-agent", type=str, choices=["dqn",
                                                     "dqn+per",
                                                     "noisy_dqn",
                                                     "noisy_dqn+per",
                                                     "dueling",
                                                     "dueling+per", 
                                                     "noisy_dueling",
                                                     "noisy_dueling+per", 
                                                     "c51",
                                                     "c51+per", 
                                                     "noisy_c51",
                                                     "noisy_c51+per", 
                                                     "duelingc51",
                                                     "duelingc51+per", 
                                                     "noisy_duelingc51",
                                                     "noisy_duelingc51+per",
                                                     "rainbow" ], default="dqn", help="Specify which type of DQN agent you want to train, default is DQN - baseline!")
    
    parser.add_argument("-env", type=str, default="PongNoFrameskip-v4", help="Name of the atari Environment, default = Pong-v0")
    parser.add_argument("-model_weights", type=str,help="Name of the saved weights")
    parser.add_argument("-eps", type=int, default=1, help="Episodes the agent shall play, default = 1")
    args = parser.parse_args()
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "-ram" in args.env or args.env == "CartPole-v0" or args.env == "LunarLander-v2": 
        env = gym.make(args.env)
    else:
        env = wrapper_new.make_env(args.env)
    seed = 1
    env.seed(seed)
    action_size     = env.action_space.n
    state_size = env.observation_space.shape

    if args.agent == "rainbow":
        args.agent = "noisy_duelingc51+per"

    if not "c51" in args.agent:
        agent = DQN_Agent(state_size=state_size,    
                        action_size=action_size,
                        Network=args.agent,
                        layer_size=512,
                        n_step=1,
                        BATCH_SIZE=32, 
                        BUFFER_SIZE=10000, 
                        LR=1, 
                        TAU=1, 
                        GAMMA=1, 
                        UPDATE_EVERY=1, 
                        device=device, 
                        seed=seed)
    else:
        agent = DQN_C51Agent(state_size=state_size,
                        action_size=action_size,
                        Network=args.agent, 
                        layer_size=512,
                        n_step=1,
                        BATCH_SIZE=11111, 
                        BUFFER_SIZE=1111, 
                        LR=1, 
                        TAU=1, 
                        GAMMA=1, 
                        UPDATE_EVERY=1, 
                        device=device, 
                        seed=seed)

    agent.qnetwork_local.load_state_dict(torch.load(args.model_weights))
    play_atari(args.eps)
