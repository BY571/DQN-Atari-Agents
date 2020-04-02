from Agents.dqn_agent import DQN_Agent
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse 
import time
import cv2

def preprocess_img(state):
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84,110))
    cropped = resized[18:102,:]
    cropped = cropped.reshape((1,84,84))
    return cropped



def run(n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.99):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        state = preprocess_img(state)
        state = np.vstack([state, state, state, state])

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            next_state = preprocess_img(next_state)
            state = state[:3,:,:]
            state = np.concatenate((new_state, state))
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-agent", type=str, choices=["dqn, ddqn, duel"], default="dqn", help="Specify which type of DQN agent you want to train, default is DQN - baseline!")
    parser.add_argument("-env", type=str, default="Pong-v0", help="Name of the atari Environment, default = Pong-v0")
    parser.add_argument("-eps", type=int, default=500, help="Number of Episodes to train, default = 500")
    parser.add_argument("-seed", type=int, default=1, help="Random seed to replicate training runs, default = 1")
    parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size for updating the DQN, default = 256")
    parser.add_argument("-m", "--memory_size", type=int, default=int(1e6), help="Replay memory size, default = 1e6")
    parser.add_argument("-u", "--update_every", type=int, default=1, help="Update the network every x steps, default = 1")
    parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate, default = 1e-3")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount factor gamma, default = 0.99")
    parser.add_argument("-t", "--tau", type=float, default=1e-3, help="Soft update parameter tat, default = 1e-3")
    parser.add_argument("-ep_start", type=float, default=1.0, help="Epsilon greedy starting value, default = 1")
    parser.add_argument("-ep_decay", type=float, default=0.99, help="Epsilon greedy decay value, default = 0.99")
    parser.add_argument("-ep_end", type=float, default = 0.05, help="Final epsilon greedy value, default = 0.05")
    parser.add_argument("-info", type=str, help="Name of the training run")
    parser.add_argument("-save_mode", type=int, choices=[0,1], default=0, help="Specify if the trained network shall be saved or not, default is 0 - not saved!")

    args = parser.parse_args()
    writer = SummaryWriter("runs/"+str(args.info))

    BUFFER_SIZE = args.memory_size
    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    TAU = args.tau
    LR = args.lr
    UPDATE_EVERY = args.update_every
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    seed = args.seed

    env = gym.make(args.env)
    env.seed(seed)
    action_size = env.action_space.n
    state_size = (4,84,84)

    agent = DQN_Agent(state_size=state_size, action_size=action_size, BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE, LR=LR, TAU=TAU, GAMMA=GAMMA,UPDATE_EVERY=UPDATE_EVERY, device=device, seed=seed)
    t0 = time.time()
    scores = run(n_episodes = args.eps, eps_start=args.ep_start, eps_end=args.ep_end, eps_decay=args.ep_decay)
    t1 = time.time()