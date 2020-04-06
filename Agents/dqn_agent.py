from .ReplayMemories.ReplayMemory import ReplayBuffer
from .Networks import DQN

import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import random


class DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 Network,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 device,
                 seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            Network (str): dqn network type
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0

        if Network == "noisy_dqn" or "noisy_dueling": 
            self.noisy = True
        else: self.noisy = False

        self.action_step = 4
        self.last_action = None

	    # Q-Network
        if Network == "noisy_dqn":
            self.qnetwork_local = DQN.DDQN(state_size, action_size, seed, layer_type="noisy").to(device)
            self.qnetwork_target = DQN.DDQN(state_size, action_size, seed, layer_type="noisy").to(device)
        if Network == "dqn":
                self.qnetwork_local = DQN.DDQN(state_size, action_size, seed).to(device)
                self.qnetwork_target = DQN.DDQN(state_size, action_size, seed).to(device)
        if Network == "noisy_dueling":
            self.qnetwork_local = DQN.Dueling_QNetwork(state_size, action_size, seed, layer_type="noisy").to(device)
            self.qnetwork_target = DQN.Dueling_QNetwork(state_size, action_size, seed, layer_type="noisy").to(device)
        if Network == "dueling":
                self.qnetwork_local = DQN.Dueling_QNetwork(state_size, action_size, seed).to(device)
                self.qnetwork_target = DQN.Dueling_QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR, alpha=0.95, eps=0.01)
        print(self.qnetwork_local)
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, writer):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                self.Q_updates += 1
                writer.add_scalar("Q_loss", loss, self.Q_updates)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy. Acting only every 4 frames!
        
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
            
        """

        if self.action_step == 4:
            state = np.array(state)

            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # Epsilon-greedy action selection
            if random.random() > eps or self.noisy: # select greedy action if random number is higher than epsilon or noisy network is used!
                action = np.argmax(action_values.cpu().data.numpy())
                self.last_action = action
                return action
            else:
                action = random.choice(np.arange(self.action_size))
                self.last_action = action 
                return action
            #self.action_step = 0
        else:
            self.action_step += 1
            return self.last_action

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Compute loss
        loss = F.smooth_l1_loss(Q_expected, Q_targets) #mse_loss
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()            

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

class DQN_C51Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 Network,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 device,
                 seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            Network (str): dqn network type
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0

        self.N_ATOMS = 51
        self.VMAX = 10
        self.VMIN = -10
        

        if Network == "noisy_dqn" or "noisy_dueling": 
            self.noisy = True
        else: self.noisy = False

        self.action_step = 4
        self.last_action = None

	    # Q-Network
        if Network == "noisy_c51":
            self.qnetwork_local = DQN.DDQN_C51(state_size, action_size, seed, layer_type="noisy").to(device)
            self.qnetwork_target = DQN.DDQN_C51(state_size, action_size, seed, layer_type="noisy").to(device)
        if Network == "c51":
                self.qnetwork_local = DQN.DDQN_C51(state_size, action_size, seed).to(device)
                self.qnetwork_target = DQN.DDQN_C51(state_size, action_size, seed).to(device)
        if Network == "noisy_duelingc51":
            self.qnetwork_local = DQN.Dueling_C51Network(state_size, action_size, seed, layer_type="noisy").to(device)
            self.qnetwork_target = DQN.Dueling_C51Network(state_size, action_size, seed, layer_type="noisy").to(device)
        if Network == "duelingc51":
                self.qnetwork_local = DQN.Dueling_C51Network(state_size, action_size, seed).to(device)
                self.qnetwork_target = DQN.Dueling_C51Network(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)#, alpha=0.95, eps=0.01)
        print(self.qnetwork_local)
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def projection_distribution(self, next_distr, next_state, rewards, dones):
        """
        """
        batch_size  = next_state.size(0)
        delta_z = float(self.VMAX - self.VMIN) / (self.N_ATOMS - 1)
        support = torch.linspace(self.VMIN, self.VMAX, self.N_ATOMS)
        rewards = rewards.expand_as(next_distr)
        dones   = dones.expand_as(next_distr)
        support = support.unsqueeze(0).expand_as(next_distr).to(self.device)
        ## Compute the projection of T̂ z onto the support {z_i}
        Tz = rewards + (1 - dones) * self.GAMMA * support
        Tz = Tz.clamp(min=self.VMIN, max=self.VMAX)
        b  = ((Tz - self.VMIN) / delta_z).cpu()#.to(self.device)
        l  = b.floor().long().cpu()#.to(self.device)
        u  = b.ceil().long().cpu()#.to(self.device)

        offset = torch.linspace(0, (batch_size - 1) * self.N_ATOMS, batch_size).long()\
                        .unsqueeze(1).expand(batch_size, self.N_ATOMS)
        # Distribute probability of T̂ z
        proj_dist = torch.zeros(next_distr.size()) 
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_distr.cpu() * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_distr.cpu() * (b - l.float())).view(-1))

        return proj_dist
    
    def step(self, state, action, reward, next_state, done, writer):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences) 
                self.Q_updates += 1
                writer.add_scalar("Q_loss", loss, self.Q_updates)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy. Acting only every 4 frames!
        
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
            
        """

        if self.action_step == 4:
            state = np.array(state)

            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.act(state)
            self.qnetwork_local.train()

            # Epsilon-greedy action selection
            if random.random() > eps or self.noisy: # select greedy action if random number is higher than epsilon or noisy network is used!
                action = np.argmax(action_values.cpu().data.numpy())
                self.last_action = action
                return action
            else:
                action = random.choice(np.arange(self.action_size))
                self.last_action = action 
                return action
            #self.action_step = 0
        else:
            self.action_step += 1
            return self.last_action

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)
        self.optimizer.zero_grad()
        # next_state distribution
        next_distr = self.qnetwork_target(next_states)
        next_actions = self.qnetwork_target.act(next_states)
        #chose max action indx
        next_actions = next_actions.max(1)[1].data.cpu().numpy()
        # gather best distr
        next_best_distr = next_distr[range(batch_size), next_actions]

        proj_distr = self.projection_distribution(next_best_distr, next_states, rewards, dones).to(self.device)

        # Compute loss
        # calculates the prob_distribution for the actions based on the given state
        prob_distr = self.qnetwork_local(states)
        actions = actions.unsqueeze(1).expand(batch_size, 1, self.N_ATOMS)
        # gathers the the prob_distribution for the chosen action
        state_action_prob = prob_distr.gather(1, actions).squeeze(1)
        loss = -(state_action_prob.log() * proj_distr.detach()).sum(dim=1).mean()
       
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()                   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)