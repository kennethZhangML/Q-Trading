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

env = gym.make("TradingEnv-v0", ticker = "AAPL")

INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = 512 
OUTPUT_DIM = env.action_space.n

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
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

def kaiming_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

def update_policy(policy, state, action, reward, next_state, discount_factor, optimizer):
    q_preds = policy(state)
    q_vals = q_preds[:, action]

    with torch.no_grad():
        q_next_preds = policy(next_state)
        q_next_vals = q_next_preds.max(1).values
        targets = reward + q_next_vals * discount_factor
    
    loss = F.smooth_l1_loss(targets.detach(), q_vals)
    nn.utils.clip_grad_norm_(policy.parameters(), max_norm = 0.5)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train(env, policy, optimizer, discount_factor, epsilon, device):
    policy.train()

    done = False 
    episode_reward = 0

    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    while not done:
        # Use epsilon-greedy exploration/exploitation strategy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_pred = policy(state)
            action = torch.argmax(q_pred).item()
        
        next_state, reward, done, info  = env.step(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        state = next_state
        
        episode_reward += reward 
        loss = update_policy(policy, state, action, reward, next_state, discount_factor, optimizer)

    return loss, episode_reward, epsilon

def evaluate(env, policy, device):
    policy.eval()

    done = False
    episode_reward = 0

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_pred = policy(state)
            action = torch.argmax(q_pred).item()

        # Important to add the info tracking variable 
        state, reward, done, info, _ = env.step(action)
        episode_reward += reward 

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
    return episode_reward


if __name__ == "__main__":
    n_runs = 10
    n_episodes = 500
    discount_factor = 0.8
    start_epsilon = 1.0
    end_epsilon = 0.01
    epsilon_decay = 0.995

    train_rewards = torch.zeros(n_runs, n_episodes)
    test_rewards = torch.zeros(n_runs, n_episodes)
    device = torch.device('cpu')

    for run in range(n_runs):
        policy = MLP(env.observation_space.shape[0], HIDDEN_DIM, OUTPUT_DIM)
        policy = policy.to(device)
        policy.apply(kaiming_weights)
        epsilon = start_epsilon

        optimizer = torch.optim.RMSprop(policy.parameters(), lr = 1e-6)

        for episode in range(n_episodes):
            loss, train_reward, epsilon = train(env, policy, optimizer, discount_factor, epsilon, device)
            epsilon *= epsilon_decay 
            epsilon = min(epsilon, end_epsilon)

            test_reward = evaluate(env, policy, device)

            train_reward[run][episode] = train_rewards 
            test_rewards[run][episode] = test_rewards 
            
        print(f"Run: {run} -> Train Rewards: {train_rewards[run]} -> Test Rewards: {test_rewards[run]}")
