import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DQN, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.cnn_1 = nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4)
        self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        self.ff_1 = nn.Linear(self.calc_input_layer(), 512)
        self.ff_2 = nn.Linear(512, action_size)
        
    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        return x.flatten().shape[0]
    
    def forward(self, input):
        """
        
        """
        x = torch.relu(self.cnn_1(input))
        x = torch.relu(self.cnn_2(x))
        x = torch.relu(self.cnn_3(x))
        x = x.view(input.size(0), -1)
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out