from .ReplayMemories.ReplayMemory import ReplayBuffer, PrioritizedReplay
from .Networks import DQN
from .IntrinsicCuriosityModule import ICM, Inverse, Forward

import numpy as np
import torch
import torch.nn
from torch.nn import KLDivLoss
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
                 layer_size,
                 n_step,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 curiosity,
                 worker,
                 device,
                 seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            Network (str): dqn network type
            layer_size (int): size of the hidden layer
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
        self.curiosity = curiosity
        self.eta = 1
        self.seed = random.seed(seed)
        self.t_seed = torch.manual_seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = 1
        self.worker = worker 
        self.BATCH_SIZE = BATCH_SIZE*worker
        self.Q_updates = 0
        self.n_step = n_step

        if "per" in Network:
            self.per = True
            Network = Network.strip("+per")
        else:
            self.per = False

        if "noisy" in Network: 
            self.noisy = True
        else: self.noisy = False
        print("Is noisy: ", self.noisy)
        self.action_step = 4
        self.last_action = None

	    # Q-Network
        if Network == "noisy_dqn":
            self.qnetwork_local = DQN.DDQN(state_size, action_size, layer_size, n_step, seed, layer_type="noisy").to(device)
            self.qnetwork_target = DQN.DDQN(state_size, action_size, layer_size, seed, layer_type="noisy").to(device)
        if Network == "dqn":
                self.qnetwork_local = DQN.DDQN(state_size, action_size,layer_size, n_step, seed).to(device)
                self.qnetwork_target = DQN.DDQN(state_size, action_size,layer_size, n_step, seed).to(device)
        if Network == "noisy_dueling":
            self.qnetwork_local = DQN.Dueling_QNetwork(state_size, action_size,layer_size, n_step, seed, layer_type="noisy").to(device)
            self.qnetwork_target = DQN.Dueling_QNetwork(state_size, action_size,layer_size, n_step, seed, layer_type="noisy").to(device)
        if Network == "dueling":
                self.qnetwork_local = DQN.Dueling_QNetwork(state_size, action_size,layer_size, n_step, seed).to(device)
                self.qnetwork_target = DQN.Dueling_QNetwork(state_size, action_size,layer_size, n_step, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)
        
        # Replay memory
        if self.per == True:
            self.memory = PrioritizedReplay(BUFFER_SIZE, BATCH_SIZE, seed=seed, gamma=self.GAMMA, n_step=n_step, parallel_env=self.worker)
        else:
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA, n_step, parallel_env=self.worker)
        
        if self.curiosity != 0:
            inverse_m = Inverse(self.state_size, self.action_size)
            forward_m = Forward(self.state_size, self.action_size, inverse_m.calc_input_layer(), device=device)
            self.ICM = ICM(inverse_m, forward_m).to(device)
            print(inverse_m, forward_m)

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
                if self.per == False:
                    loss, icm_loss = self.learn(experiences)
                else: 
                    loss = self.learn_per(experiences)
                self.Q_updates += 1
                writer.add_scalar("Q_loss", loss, self.Q_updates)
                writer.add_scalar("ICM_loss", icm_loss, self.Q_updates)

    def act(self, state, eps=0., eval=False):
        """Returns actions for given state as per current policy. Acting only every 4 frames!
        
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
            
        """

        # Epsilon-greedy action selection
        if random.random() > eps: # select greedy action if random number is higher than epsilon or noisy network is used!
            state = np.array(state)
            if len(self.state_size) > 1:
                state = torch.from_numpy(state).float().to(self.device)        
            else:
                state = torch.from_numpy(state).float().to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
            return action
        else:
            if eval:
                action = random.choices(np.arange(self.action_size), k=1)
            else:
                action = random.choices(np.arange(self.action_size), k=self.worker)
            return action

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        # calculate curiosity
        if self.curiosity != 0:
            forward_pred_err, inverse_pred_err = self.ICM.calc_errors(state1=states, state2=next_states, action=actions)
            r_i = self.eta * forward_pred_err
            assert r_i.shape == rewards.shape, "r_ and r_e have not the same shape"
            if self.curiosity == 1:
                rewards += r_i.detach()
            else:
                rewards = r_i.detach()

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.GAMMA**self.n_step * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)



        if self.curiosity != 0:
            icm_loss = self.ICM.update_ICM(forward_pred_err, inverse_pred_err)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) 
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy(),  icm_loss            

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

    def learn_per(self, experiences):
            """Update value parameters using given batch of experience tuples.
            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            self.optimizer.zero_grad()
            states, actions, rewards, next_states, dones, idx, weights = experiences
            
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
            dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states 
            Q_targets = rewards + (self.GAMMA**self.n_step * Q_targets_next * (1 - dones))
            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)
            # Compute loss
            td_error =  Q_targets - Q_expected
            loss = td_error.pow(2)*weights.mean().to(self.device)
            # Minimize the loss
            loss.backward()
            clip_grad_norm_(self.qnetwork_local.parameters(),1)
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target)
            # update per priorities
            self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))

            return loss.detach().cpu().numpy()            


class DQN_C51Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 Network,
                 layer_size,
                 n_step,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 worker,
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
        self.t_seed = torch.manual_seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = 1
        self.worker = worker
        self.BATCH_SIZE = BATCH_SIZE*worker
        self.Q_updates = 0
        self.Network = Network
        self.n_step = n_step

        self.N_ATOMS = 51
        self.VMAX = 10
        self.VMIN = -10
        
        if "per" in Network:
            self.per = True
            Network = Network.strip("+per")
        else:
            self.per = False

        if "noisy" in Network:
            self.noisy = True
        else: self.noisy = False

        print("Is noisy: ", self.noisy)
        self.action_step = 4
        self.last_action = None

	    # Q-Network
        if Network == "noisy_c51":
            self.qnetwork_local = DQN.DDQN_C51(state_size, action_size,layer_size, n_step, seed, layer_type="noisy").to(device)
            self.qnetwork_target = DQN.DDQN_C51(state_size, action_size,layer_size, n_step, seed, layer_type="noisy").to(device)
        if Network == "c51":
                self.qnetwork_local = DQN.DDQN_C51(state_size, action_size,layer_size, n_step, seed).to(device)
                self.qnetwork_target = DQN.DDQN_C51(state_size, action_size,layer_size, n_step, seed).to(device)
        if Network == "noisy_duelingc51":
            self.qnetwork_local = DQN.Dueling_C51Network(state_size, action_size,layer_size, n_step, seed, layer_type="noisy").to(device)
            self.qnetwork_target = DQN.Dueling_C51Network(state_size, action_size,layer_size, n_step, seed, layer_type="noisy").to(device)
        if Network == "duelingc51":
                self.qnetwork_local = DQN.Dueling_C51Network(state_size, action_size,layer_size, n_step, seed).to(device)
                self.qnetwork_target = DQN.Dueling_C51Network(state_size, action_size,layer_size, n_step, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)
        # Replay memory
        if self.per == True:
            self.memory = PrioritizedReplay(BUFFER_SIZE, BATCH_SIZE, seed=seed, gamma=self.GAMMA, n_step=n_step, parallel_env=self.worker)
        else:
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed,self.GAMMA, n_step, parallel_env=self.worker)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def projection_distribution(self, next_distr, next_state, rewards, dones):
        """
        """
        batch_size  = next_state.size(0)
        # create support atoms
        delta_z = float(self.VMAX - self.VMIN) / (self.N_ATOMS - 1)
        support = torch.linspace(self.VMIN, self.VMAX, self.N_ATOMS)
        support = support.unsqueeze(0).expand_as(next_distr).to(self.device)

        rewards = rewards.expand_as(next_distr)
        dones   = dones.expand_as(next_distr)
        
        ## Compute the projection of T̂ z onto the support {z_i}
        Tz = rewards + (1 - dones) * self.GAMMA**self.n_step * support
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
                if self.per == False:
                    loss = self.learn(experiences) 
                else:
                    loss = self.learn_per(experiences) 
                self.Q_updates += 1
                writer.add_scalar("Q_loss", loss, self.Q_updates)

    def act(self, state, eps=0., eval=False):
        """Returns actions for given state as per current policy. Acting only every 4 frames!
        
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
            
        """

        # Epsilon-greedy action selection
        if random.random() > eps: # select greedy action if random number is higher than epsilon or noisy network is used!
            state = np.array(state)
            if len(self.state_size) > 1:
                state = torch.from_numpy(state).float().to(self.device)        
            else:
                state = torch.from_numpy(state).float().to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
            return action
        else:
            if eval:
                action = random.choices(np.arange(self.action_size), k=1)
            else:
                action = random.choices(np.arange(self.action_size), k=self.worker)
            return action

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

    def learn_per(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idx, weights = experiences

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        batch_size = self.BATCH_SIZE
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
        loss_prio = -((state_action_prob.log() * proj_distr.detach()).sum(dim=1).unsqueeze(1)*weights) # at some point none values arise
        #print("LOSS: ",loss_prio)
        loss = loss_prio.mean()

        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        # update per priorities
        self.memory.update_priorities(idx, abs(loss_prio.data.cpu().numpy()))
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
