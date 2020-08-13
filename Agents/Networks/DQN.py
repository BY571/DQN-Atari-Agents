import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class NoisyLinear(nn.Linear):
    # Noisy Linear Layer for independent Gaussian Noise
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # not trainable tensor for the nn.Module
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        # extra parameter for the bias and register buffer for the bias parameter
        if bias: 
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
    
        # reset parameter as initialization of the layer
        self.reset_parameter()
    
    def reset_parameter(self):
        """
        initialize the parameter of the layer and bias
        """
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    
    def forward(self, input):
        # sample random noise in sigma weight buffer and bias buffer
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)

class DDQN(nn.Module):
    def __init__(self, state_size, action_size,layer_size, n_step, seed, layer_type="ff"):
        super(DDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.state_dim = len(state_size)
        if self.state_dim == 3:
            self.cnn_1 = nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4)
            self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
            weight_init([self.cnn_1, self.cnn_2, self.cnn_3])

            if layer_type == "noisy":
                self.ff_1 = NoisyLinear(self.calc_input_layer(), layer_size)
                self.ff_2 = NoisyLinear(layer_size, action_size)
            else:
                self.ff_1 = nn.Linear(self.calc_input_layer(), layer_size)
                self.ff_2 = nn.Linear(layer_size, action_size)
                weight_init([self.ff_1])
        elif self.state_dim == 1:
            if layer_type == "noisy":
                self.head_1 = nn.Linear(self.input_shape[0], layer_size)
                self.ff_1 = NoisyLinear(layer_size, layer_size)
                self.ff_2 = NoisyLinear(layer_size, action_size)
            else:
                self.head_1 = nn.Linear(self.input_shape[0], layer_size)
                self.ff_1 = nn.Linear(layer_size, layer_size)
                self.ff_2 = nn.Linear(layer_size, action_size)
                weight_init([self.head_1, self.ff_1])
        else:
            print("Unknown input dimension!")


        
    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        return x.flatten().shape[0]
    
    def forward(self, input):
        """
        
        """
        if self.state_dim == 3:
            x = torch.relu(self.cnn_1(input))
            x = torch.relu(self.cnn_2(x))
            x = torch.relu(self.cnn_3(x))
            x = x.view(input.size(0), -1)
        else:
            x = torch.relu(self.head_1(input))
        
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out

class Dueling_QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size,layer_size, n_step, seed, layer_type="ff"):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Dueling_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.state_dim = len(self.input_shape)
        self.action_size = action_size
        if self.state_dim == 3:
            self.cnn_1 = nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4)
            self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
            weight_init([self.cnn_1, self.cnn_2, self.cnn_3])
            if layer_type == "noisy":
                self.ff_1_A = NoisyLinear(self.calc_input_layer(), layer_size)
                self.ff_1_V = NoisyLinear(self.calc_input_layer(), layer_size)
                self.advantage = NoisyLinear(layer_size,action_size)
                self.value = NoisyLinear(layer_size,1)
                weight_init([self.ff_1_A, self.ff_1_V])
            else:
                self.ff_1_A = nn.Linear(self.calc_input_layer(), layer_size)
                self.ff_1_V = nn.Linear(self.calc_input_layer(), layer_size)
                self.advantage = nn.Linear(layer_size,action_size)
                self.value = nn.Linear(layer_size,1)
                weight_init([self.ff_1_A, self.ff_1_V])
        elif self.state_dim == 1:
            if layer_type == "noisy":
                self.head_1 = nn.Linear(self.input_shape[0], layer_size)
                self.ff_1_A = NoisyLinear(layer_size, layer_size)
                self.ff_1_V = NoisyLinear(layer_size, layer_size)
                self.advantage = NoisyLinear(layer_size,action_size)
                self.value = NoisyLinear(layer_size,1)
                weight_init([self.head_1,self.ff_1_A, self.ff_1_V])
            else:
                self.head_1 = nn.Linear(self.input_shape[0], layer_size)
                self.ff_1_A = nn.Linear(layer_size, layer_size)
                self.ff_1_V = nn.Linear(layer_size, layer_size)
                self.advantage = nn.Linear(layer_size,action_size)
                self.value = nn.Linear(layer_size,1)
                weight_init([self.head_1,self.ff_1_A, self.ff_1_V])
        else:
            print("Unknown input dimension!")

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        return x.flatten().shape[0]

    def forward(self, input):
        """
        """
        if self.state_dim == 3:
            x = torch.relu(self.cnn_1(input))
            x = torch.relu(self.cnn_2(x))
            x = torch.relu(self.cnn_3(x))
            x = x.view(input.size(0), -1)
            x_A = torch.relu(self.ff_1_A(x))
            x_V = torch.relu(self.ff_1_V(x))   
        else:
            x = torch.relu(self.head_1(input))
            x_A = torch.relu(self.ff_1_A(x))
            x_V = torch.relu(self.ff_1_V(x))

        value = self.value(x_V)
        value = value.expand(input.size(0), self.action_size)
        advantage = self.advantage(x_A)
        Q = value + advantage - advantage.mean()
        return Q

class Dueling_C51Network(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size,layer_size, n_step, seed, layer_type="ff", N_ATOMS=51, VMAX=10, VMIN=-10):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Dueling_C51Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.state_dim = len(self.input_shape)
        self.action_size = action_size
        self.N_ATOMS = N_ATOMS
        self.VMAX = VMAX
        self.VMIN = VMIN
        self.DZ = (VMAX-VMIN) / (N_ATOMS - 1)


        if self.state_dim == 3:
            self.cnn_1 = nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4)
            self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
            weight_init([self.cnn_1, self.cnn_2, self.cnn_3])

            if layer_type == "noisy":
                self.ff_1_A = NoisyLinear(self.calc_input_layer(), layer_size)
                self.ff_1_V = NoisyLinear(self.calc_input_layer(), layer_size)
                self.advantage = NoisyLinear(layer_size,action_size*N_ATOMS)
                self.value = NoisyLinear(layer_size,N_ATOMS)
                weight_init([self.ff_1_A, self.ff_1_V])
            else:
                self.ff_1_A = nn.Linear(self.calc_input_layer(), layer_size)
                self.ff_1_V = nn.Linear(self.calc_input_layer(), layer_size)
                self.advantage = nn.Linear(layer_size,action_size*N_ATOMS)
                self.value = nn.Linear(layer_size,N_ATOMS)
                weight_init([self.ff_1_A, self.ff_1_V])
        elif self.state_dim == 1:
            if layer_type == "noisy":
                self.head_1 = nn.Linear(self.input_shape[0], layer_size)
                self.ff_1_A = NoisyLinear(layer_size, layer_size)
                self.ff_1_V = NoisyLinear(layer_size, layer_size)
                self.advantage = NoisyLinear(layer_size,action_size*N_ATOMS)
                self.value = NoisyLinear(layer_size,N_ATOMS)
                weight_init([self.head_1,self.ff_1_A, self.ff_1_V])
            else:
                self.head_1 = nn.Linear(self.input_shape[0], layer_size)
                self.ff_1_A = nn.Linear(layer_size, layer_size)
                self.ff_1_V = nn.Linear(layer_size, layer_size)
                self.advantage = nn.Linear(layer_size,action_size*N_ATOMS)
                self.value = nn.Linear(layer_size,N_ATOMS)
                weight_init([self.head_1,self.ff_1_A, self.ff_1_V])
        else:
            print("Unknown input dimension!")

        self.register_buffer("supports", torch.arange(VMIN, VMAX+self.DZ, self.DZ)) # basic value vector - shape n_atoms stepsize dz
        self.softmax = nn.Softmax(dim = 1)
        
    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        return x.flatten().shape[0]

    def forward(self, input):
        batch_size = input.size()[0]
        if self.state_dim == 3:
            x = torch.relu(self.cnn_1(input))
            x = torch.relu(self.cnn_2(x))
            x = torch.relu(self.cnn_3(x))
            x = x.view(input.size(0), -1)
            x_A = torch.relu(self.ff_1_A(x))
            x_V = torch.relu(self.ff_1_V(x))    
        else:
            x = torch.relu(self.head_1(input))
            x_A = torch.relu(self.ff_1_A(x))
            x_V = torch.relu(self.ff_1_V(x)) 

        value = self.value(x_V).view(batch_size,1,self.N_ATOMS)
        advantage = self.advantage(x_A).view(batch_size,-1, self.N_ATOMS)
        q_distr = value + advantage - advantage.mean(dim = 1, keepdim = True)
        prob = self.softmax(q_distr.view(-1, self.N_ATOMS)).view(-1, self.action_size, self.N_ATOMS)
        return prob
      
    def act(self,state):
      prob = self.forward(state).data.cpu()
      expected_value = prob.cpu() * self.supports.cpu()
      actions = expected_value.sum(2)
      return actions

class DDQN_C51(nn.Module):
    def __init__(self, state_size, action_size,layer_size, n_step, seed, layer_type="ff", N_ATOMS=51, VMAX=10, VMIN=-10):
        super(DDQN_C51, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.state_dim = len(state_size)
        self.N_ATOMS = N_ATOMS
        self.VMAX = VMAX
        self.VMIN = VMIN
        self.DZ = (VMAX-VMIN) / (N_ATOMS - 1)
        
        if self.state_dim == 3:
            self.cnn_1 = nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4)
            self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
            weight_init([self.cnn_1, self.cnn_2, self.cnn_3])

            if layer_type == "noisy":
                self.ff_1 = NoisyLinear(self.calc_input_layer(), layer_size)
                self.ff_2 = NoisyLinear(layer_size, action_size*N_ATOMS)
            else:
                self.ff_1 = nn.Linear(self.calc_input_layer(), layer_size)
                self.ff_2 = nn.Linear(layer_size, action_size*N_ATOMS)
                weight_init([self.ff_1])
        elif self.state_dim == 1:
            if layer_type == "noisy":
                self.head_1 = nn.Linear(self.input_shape[0], layer_size)
                self.ff_1 = NoisyLinear(layer_size, layer_size)
                self.ff_2 = NoisyLinear(layer_size, action_size*N_ATOMS)
            else:
                self.head_1 = nn.Linear(self.input_shape[0], layer_size)
                self.ff_1 = nn.Linear(layer_size, layer_size)
                self.ff_2 = nn.Linear(layer_size, action_size*N_ATOMS)
                weight_init([self.head_1, self.ff_1])
        else:
            print("Unknown input dimension!")
        self.register_buffer("supports", torch.arange(VMIN, VMAX+self.DZ, self.DZ)) # basic value vector - shape n_atoms stepsize dz
        self.softmax = nn.Softmax(dim = 1)


        
    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        return x.flatten().shape[0]
    
    def forward(self, input):
        batch_size = input.size()[0]
        if self.state_dim == 3:
            x = torch.relu(self.cnn_1(input))
            x = torch.relu(self.cnn_2(x))
            x = torch.relu(self.cnn_3(x))
            x = x.view(input.size(0), -1)
            x = torch.relu(self.ff_1(x))
  
        else:
            x = torch.relu(self.head_1(input))
            x = torch.relu(self.ff_1(x))  

        
        q_distr = self.ff_2(x)
        prob = self.softmax(q_distr.view(-1, self.N_ATOMS)).view(-1, self.action_size, self.N_ATOMS)
        return prob
      
    def act(self,state):
      prob = self.forward(state).data.cpu()
      # create value distribution for each action - shape: (batch_size, action_space, 51)
      expected_value = prob.cpu() * self.supports.cpu()
      # sum up the prob*values for the action dimension - shape: (batch_size, action_space)
      actions = expected_value.sum(2)
      return actions
