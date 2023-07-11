import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import gym 
import yfinance
import numpy as np 
from tradingEnv_portfolioWeighted import TradingEnv


gym.envs.register(
    id = "TradingEnv-v0",
    entry_point = "tradingEnv:TradingEnv"
)

stock_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
portfolio_weights = [0.3, 0.2, 0.3, 0.2]

start_date = input("Enter a start_date (YYYY-MM-DD): ")
end_date = input("Enter an end date (YYYY-MM-DD): ")

import gym
import numpy as np

# Q-Learning Parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000

env = TradingEnv(stock_tickers, portfolio_weights, start_date, end_date)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
q_table = {}

for episode in range(num_episodes):
    state = env.reset()
    state_str = str(state) 
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  
        else:
            if state_str in q_table:
                action = np.argmax(q_table[state_str])
            else:
                action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        next_state_str = str(next_state) 

        if state_str not in q_table:
            q_table[state_str] = np.zeros(num_actions)
        if next_state_str not in q_table:
            q_table[next_state_str] = np.zeros(num_actions)

        q_value = q_table[state_str][action]
        max_q_value = np.max(q_table[next_state_str])

        if episode % 100 == 0:
            print(f"Episode: {episode} -> Reward: {reward}")

state = env.reset()
state_str = str(state)
done = False 

while not done:
    if state_str in q_table:
        action = np.argmax(q_table[state_str])
    else:
        action = env.action_space.sample()
    
    state, reward, done, _ = env.step(action)
    state_str = str(state)
    env.render()

env.close()
