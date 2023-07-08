import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 
import pandas as pd 
import gym

from tradingEnv import TradingEnv

gym.envs.register(
    id = "TradingEnv-v0",
    entry_point = "tradingEnv:TradingEnv"
)

env = gym.make("TradingEnv-v0")

INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = 512 
OUTPUT_DIm = env.action_space.n

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
env.seed(SEED)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x 

# Work in progress...