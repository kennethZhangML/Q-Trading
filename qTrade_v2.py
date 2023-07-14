import gym 
import numpy as np 

from tradingEnv_portfolioWeighted import *

stock_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
portfolio_weights = [0.3, 0.2, 0.3, 0.2]

# start_date = input("Enter the start date (YYYY-MM-DD): ")
# end_date = input("Enter the end date (YYYY-MM-DD): ")
start_date = "2022-01-01"
end_date = "2023-01-01"

alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 1000

env = TradingEnv(stock_tickers, portfolio_weights, start_date, end_date)

states = env.observation_space.shape[0]
actions = env.action_space.shape[0]
q_table = np.zeros((states, actions))

for episode in range(num_episodes):
    state = env.reset()
    state_tuple = tuple(map(tuple, state))
    done = False 

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[int(state)])
        
        next_state, reward, done, _ = env.step(action)
        next_state_tuple = tuple(map(tuple, next_state))
        q_table[state, action] = (1 - alpha) * q_table[state_tuple, action] \
                            + alpha * (reward + gamma * np.max(q_table[next_state_tuple]))
    
        state = next_state 

    print(f"Episode: {episode + 1}, Total Reward: {np.sum(env.rewards)}")

state = env.reset()
state_tuple = tuple(map(tuple, state))
done = False 
while not done:
    action = np.argmax(q_table[state_tuple])
    state, _, done, _ = env.step(action)
    env.render()

env.close()
        