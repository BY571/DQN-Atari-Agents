from Agents.dqn_agent import DQN_Agent
from Wrapper import wrapper
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
import matplotlib.pyplot as plt

def run_random_policy(random_frames):
    """
    Run env with random policy for x frames to fill the replay memory.
    """
    state = env.reset()
    action_translator = {0:0, 1:2, 2:3} 
    for i in range(random_frames):
        action = np.random.randint(action_size)  #env.action_space.sample()
        #action_ = action_translator[action]
        next_state, reward, done, _ = env.step(action)#_)
        agent.memory.add(state, action, reward, next_state, done)
        next_state = state
        if done:
            state = env.reset()

def run(frames=1000, eps_frames=1e6, min_eps=0.01):
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
    frame = 0
    eps = 1
    eps_start = 1
    action_translator = {0:0, 1:2, 2:3}
    i_episode = 1
    state = env.reset()
    score = 0                  
    for frame in range(1, frames+1):

        action = agent.act(state, eps)
        #action_ = action_translator[action.item()]
        next_state, reward, done, _ = env.step(action) #_
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if frame < eps_frames:
            eps = max(eps_start - (frame*(1/eps_frames)), min_eps)
        #else:
        #    eps = min_eps - ((frame-eps_frames)/(frames-eps_frames))

        if done:
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            writer.add_scalar("Epsilon", eps, i_episode)
            writer.add_scalar("Reward", score, i_episode)
            writer.add_scalar("Average100", np.mean(scores_window), i_episode)
            print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f}'.format(i_episode, frame, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode,frame, np.mean(scores_window)))
            i_episode +=1 
            state = env.reset()
            score = 0              

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-agent", type=str, choices=["dqn", "noisy_dqn", "dueling", "noisy_dueling" ], default="dqn", help="Specify which type of DQN agent you want to train, default is DQN - baseline!")
    parser.add_argument("-env", type=str, default="PongDeterministic-v4", help="Name of the atari Environment, default = Pong-v0")
    parser.add_argument("-frames", type=int, default=int(5e6), help="Number of frames to train, default = 5 mio")
    parser.add_argument("-seed", type=int, default=1, help="Random seed to replicate training runs, default = 1")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size for updating the DQN, default = 32")
    parser.add_argument("-m", "--memory_size", type=int, default=int(1e6), help="Replay memory size, default = 1e6")
    parser.add_argument("-u", "--update_every", type=int, default=1, help="Update the network every x steps, default = 1")
    parser.add_argument("-lr", type=float, default=5e-4, help="Learning rate, default = 5e-4")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount factor gamma, default = 0.99")
    parser.add_argument("-t", "--tau", type=float, default=1e-3, help="Soft update parameter tat, default = 1e-3")
    parser.add_argument("-eps_frames", type=int, default=1e4, help="Linear annealed frames for Epsilon, default = 1e4")
    parser.add_argument("-min_eps", type=float, default = 0.1, help="Final epsilon greedy value, default = 0.1")
    parser.add_argument("-info", type=str, help="Name of the training run")
    parser.add_argument("--fill_buffer", type=int, default=None, help="Adding samples to the replay buffer based on a random policy, before agent-env-interaction. Input numer of preadded frames to the buffer, default = None")
    parser.add_argument("-save_model", type=int, choices=[0,1], default=0, help="Specify if the trained network shall be saved or not, default is 0 - not saved!")

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
    if not "ram" in args.env and args.env != "CartPole-v0": 
        env = wrapper.wrap_deepmind(env)
    action_size = env.action_space.n # 3
    state_size = env.observation_space.shape

    agent = DQN_Agent(state_size=state_size,
                      action_size=action_size,
                      Network=args.agent, 
                      BATCH_SIZE=BATCH_SIZE, 
                      BUFFER_SIZE=BUFFER_SIZE, 
                      LR=LR, 
                      TAU=TAU, 
                      GAMMA=GAMMA, 
                      UPDATE_EVERY=UPDATE_EVERY, 
                      device=device, 
                      seed=seed)
    
    # adding x frames of random policy to the replay buffer before training!
    if args.fill_buffer != None:
        run_random_policy(args.fill_buffer)
        print("Buffer size: ", agent.memory.__len__())


    t0 = time.time()
    scores = run(frames = args.frames, eps_frames=args.eps_frames, min_eps=args.min_eps)
    t1 = time.time()
    
    print("Training time: {}min".format((t1-t0)/60))
    if args.save_model:
        torch.save(agent.qnetwork_local.state_dict(), str(args.info))