import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

class SingleFCLayer(nn.Sequential):

    def __init__(self, in_size, out_size, activation=nn.ReLU, dropout=0.5):
        super().__init__()
        self.add_module("fc", nn.Linear(in_size, out_size))
        if activation:
            self.add_module("activation", activation())
        self.add_module("dropout", nn.Dropout(dropout))

    def forward(self, V):
        return super().forward(V).squeeze()

class FCNet(nn.Sequential):
    def __init__(self, input_size, output_size, n_layers, size, 
                  activation=nn.ReLU, output_activation=None):
        super().__init__()
        in_size = input_size
        out_size = size
        for i in range(n_layers):
            self.add_module('Net'+str(i+1), SingleFCLayer(in_size, out_size, activation))
            in_size = out_size
        self.add_module('OutputNet', SingleFCLayer(in_size, output_size, output_activation, dropout=0))
    
    def forward(self, V):
        return super().forward(V).squeeze()

class CovNet(nn.Module):

    def __init__(self, input_size, output_size):
        super(CovNet, self).__init__()
        h, w, l = input_size
        assert l==3, 'input size should have 3 channels'
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))