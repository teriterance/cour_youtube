import torch 
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F

class GabinModel(nn.Module):

    def __init__(self):
        """description des couches de notre model"""
        super(GabinModel, self).__init__()
        #couche convolutionelle 
        self.conv1 = nn.Conv2d(4, 6, kernel_size = 3, stride = 1, padding =1)
        
        #maxpool
        self.pool = nn.MaxPool2d(kernel_size = 2,  stride = 2, padding = 0)
        
        #partie lineaire 
        self.fc1 = nn.Linear(6*32*32, 780)
        self.fc2 = nn.Linear(780, 64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        #dimension de x, (3, 64, 64) -> (6, 64,64)
        x = self.conv1(x)

        #dimmension de x, (6, 64,64) -> (6, 64,64)
        x = F.relu(x)

        #dimmension de x, (6, 64,64) -> (3, 32, 32)
        x = self.pool(x)

        #debut du reseau lineaire associe

        #lineariastion de x
        #dimmension de x, (3, 32, 32) -> 3*32*32
        x =  x.view(x.size(0), -1)

        #dimmension de x, 3*32*32 -> 780
        x = F.relu(self.fc1(x))

        #dimmension de x, 780 -> 64
        x = F.relu(self.fc2(x))

        #dimmension de x, 64 -> 10
        x = F.relu(self.fc3(x))
        
        return x