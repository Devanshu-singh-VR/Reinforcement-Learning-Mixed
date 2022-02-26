import torch.nn as nn
from torch.nn import Module
import torch

class DQNetwork(Module):
    def __init__(self, num_states, num_actions):
        super(DQNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc_layer1 = nn.Linear(num_states, 256)
        self.fc_layer2 = nn.Linear(256, 256)
        self.q_val = nn.Linear(256, num_actions)

    def forward(self, x):
        x = self.relu(self.fc_layer1(x))
        x = self.relu(self.fc_layer2(x))
        out = self.q_val(x)
        return out

if __name__ == '__main__':
    model = DQNetwork(8, 4)
    inputs = torch.ones((64, 8))
    print(model(inputs).shape)

