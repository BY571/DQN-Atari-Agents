import torch
import numpy as np 
import torch.nn as nn 
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Inverse(nn.Module):
    """
    1. (first submodel) encodes the state and next state into feature space.
    2. (second submodel) the inverse approximates the action taken by the given state and next state in feature size
    
    returns the predicted action and the encoded state for the Forward Model and the encoded next state to train the forward model!
    
    optimizing the Inverse model by the loss between actual action taken by the current policy and the predicted action by the inverse model
    """
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Inverse, self).__init__()
        self.state_dim = len(state_size)
        self.state_size = state_size
        
        if self.state_dim == 1:
            self.encoder = nn.Sequential(nn.Linear(state_size, 128),
                                         nn.ELU())
            self.layer1 = nn.Linear(2*128, hidden_size)
        else:
            self.encoder = nn.Sequential(nn.Conv2d(in_channels=self.state_size[0], out_channels=32, kernel_size=3, stride=2, padding=1),
                                         nn.ELU(),
                                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
                                         nn.ELU(),
                                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
                                         nn.ELU(),
                                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
                                         nn.ELU())
                                         
            self.layer1 = nn.Linear(2*self.calc_input_layer(), hidden_size)

        self.layer2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=1)
    
    def calc_input_layer(self):
        x = torch.zeros(self.state_size).unsqueeze(0)
        x = self.encoder(x)
        return x.flatten().shape[0]
    
    def forward(self, enc_state, enc_state1):
        """
        Input: state s and state s' as torch Tensors with shape: (batch_size, state_size)
        Output: action probs with shape (batch_size, action_size)
        """
        #stacked_states = torch.stack((state1,state2))
        #output = self.encoder(stacked_states)
        #enc_state = output[0].view(state.shape[0], -1)
        #enc_state1 = output[1].view(state.shape[0], -1)
        #print(enc_state1.shape)
        x = torch.cat((enc_state, enc_state1), dim=1)
        x = torch.relu(self.layer1(x))
        x = self.softmax(self.layer2(x))
        return x

    
class Forward(nn.Module):
    """
  
    """
    def __init__(self, state_size, action_size, output_size, hidden_size=256, device="cuda:0"):
        super(Forward, self).__init__()
        self.action_size = action_size
        self.device = device
        self.forwardM = nn.Sequential(nn.Linear(output_size+self.action_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size,output_size))
    
    def forward(self, state, action):
        """
        Input: state s embeddings and action a as torch Tensors with shape
        s: (batch_size, embedding_size), 
        a: (batch_size, action_size)
        
        Output:
        encoded state s' prediction by the forward model with shape: (batch_size, embedding_size)
        
        Gets as inputs the aciton taken from the policy and the encoded state by the encoder in the inverse model.
        The froward model trys to predict the encoded next state. 
        Returns the predicted encoded next state.
        Gets optimized by the MSE between the actual encoded next state and the predicted version of the forward model!

         """
        # One-hot-encoding for the actions 
        ohe_action = torch.zeros(action.shape[0], self.action_size).to(self.device)
        indices = torch.stack((torch.arange(action.shape[0]).to(self.device), action.squeeze().long()), dim=0)
        indices = indices.tolist()
        ohe_action[indices] = 1.
        #concat state embedding and encoded action

        x = torch.cat((state, ohe_action) ,dim=1)
        assert x.device.type == "cuda"
        return self.forwardM(x)

    
class ICM(nn.Module):
    def __init__(self, inverse_model, forward_model, learning_rate=1e-3, lambda_=0.1, beta=0.2, device="cuda:0"):
        super(ICM, self).__init__()
        self.inverse_model = inverse_model.to(device)
        self.forward_model = forward_model.to(device)
        
        self.forward_scale = 1.
        self.inverse_scale = 1e4
        self.lr = learning_rate
        self.beta = beta
        self.lambda_ = lambda_
        self.forward_loss = nn.MSELoss(reduction='none')
        self.inverse_loss = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.Adam(list(self.forward_model.parameters())+list(self.inverse_model.parameters()), lr=1e-3)

    def calc_errors(self, state1, state2, action):
        """
        Input: Torch Tensors state s, state s', action a with shapes
        s: (batch_size, state_size)
        s': (batch_size, state_size)
        a: (batch_size, 1)
        
        """
        assert state1.device.type == "cuda" and state2.device.type == "cuda" and action.device.type == "cuda"
        enc_state1 = self.inverse_model.encoder(state1).view(state1.shape[0],-1)
        enc_state2 = self.inverse_model.encoder(state2).view(state1.shape[0],-1)

        #assert enc_state1.shape == (32,1152), "Shape is {}".format(enc_state1.shape)
        # calc forward error 
        forward_pred = self.forward_model(enc_state1.detach(), action)
        forward_pred_err = 1/2 * self.forward_loss(forward_pred, enc_state2.detach()).sum(dim=1).unsqueeze(dim=1)
        
        # calc prediction error
        pred_action = self.inverse_model(enc_state1, enc_state2) 
        inverse_pred_err = self.inverse_loss(pred_action, action.flatten().long()).unsqueeze(dim=1)    
      
        return forward_pred_err, inverse_pred_err

    def update_ICM(self, forward_err, inverse_err):
        self.optimizer.zero_grad()
        loss = ((1. - self.beta)*inverse_err + self.beta*forward_err).mean()
        #print(loss)
        loss.backward(retain_graph=True)
        clip_grad_norm_(self.inverse_model.parameters(),1)
        clip_grad_norm_(self.forward_model.parameters(),1)
        self.optimizer.step()
        return loss.detach().cpu().numpy()