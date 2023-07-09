import gym
from gym import spaces

import yfinance as yf
import numpy as np
import pandas as pd 
import warnings

import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

warnings.filterwarnings("ignore", category = DeprecationWarning, module = "gym")
warnings.filterwarnings("ignore", category = UserWarning, module = "gym")

class TradingEnv(gym.Env):
    def __init__(self, ticker):
        super(TradingEnv, self).__init__()

        self.ticker = ticker 
        self.observation_space = spaces.Box(low = 0, high = 1, shape = (10,))
        self.action_space = spaces.Discrete(3) 

        self.current_step = 0
        self.max_steps = 1000
        self.prices = pd.DataFrame({'Close': np.random.rand(self.max_steps)}, 
                                    index = pd.date_range(start = '2023-01-01', periods = self.max_steps))
        
        self.stop_loss_level = None
        self.take_profit_level = None

        self.fig, self.ax = plt.subplots()
        self.line = self.ax.plot([], [])
        self.animation = None 

    def reset(self, **kwargs):
        self.current_step = 0
        self.stop_loss_level = None
        self.take_profit_level = None

        observation = self._get_observation()
        return observation

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps

        if not done:
            observation = self._get_observation()
            reward = self._calculate_reward(action)
        else:
            observation = None
            reward = 0.0

        return observation, reward, done, {}

    def _get_observation(self):
        start_index = self.current_step
        end_index = min(self.current_step + 10, self.max_steps)

        if end_index <= start_index:
            observation = np.zeros((10,))  
        else:
            start_date = self.prices.index[start_index].strftime('%Y-%m-%d')
            end_date = self.prices.index[end_index - 1].strftime('%Y-%m-%d')

            symbol = self.ticker  

            stock_data = yf.download(symbol, start=start_date, end=end_date)
            observation = stock_data['Close'].values

        return observation

    def _calculate_reward(self, action):
        current_price = self.prices.iloc[self.current_step]['Close']
        previous_price = self.prices.iloc[self.current_step - 1]['Close']

        price_change = current_price - previous_price

        buy_threshold = 0.02 
        sell_threshold = 0.01  
        stop_loss = -0.03 
        take_profit = 0.05  

        if action == 0 and price_change > buy_threshold:
            reward = 1.0 
            self.stop_loss_level = current_price * (1 + stop_loss)
            self.take_profit_level = current_price * (1 + take_profit)

        elif action == 1 and price_change < -sell_threshold:
            reward = -1.0 
            self.stop_loss_level = None
            self.take_profit_level = None

        elif self.stop_loss_level is not None and current_price <= self.stop_loss_level:
            reward = -1.0 
            self.stop_loss_level = None
            self.take_profit_level = None

        elif self.take_profit_level is not None and current_price >= self.take_profit_level:
            reward = 1.0  
            self.stop_loss_level = None
            self.take_profit_level = None

        else:
            reward = 0.0  

        return reward

    def render(self, mode='human'):
        if self.animation is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(self.prices.index[0], self.prices.index[-1])
            self.ax.set_ylim(np.min(self.prices['Close']), np.max(self.prices['Close']))
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax.set_title('Stock Price')

            self.line, = self.ax.plot([], [])
            self.animation = FuncAnimation(self.fig, self._animate, frames = len(self.prices), interval=100, blit=False)

            plt.show()
        else:
            self.animation.event_source.start()

    def _animate(self, i):
        current_prices = self.prices.iloc[:i+1]['Close']
        self.line.set_data(current_prices.index, current_prices.values)
        self.ax.relim()
        self.ax.autoscale_view()

    def close(self):
        pass

if __name__ == "__main__":
    gym.envs.register(
        id = 'TradingEnv-v0',
        entry_point = 'tradingEnv:TradingEnv',
    )

    user_ticker = input("Enter a ticker symbol: ")

    env = gym.make('TradingEnv-v0', ticker = user_ticker, new_step_api = True)
    observation = env.reset()

    for _ in range(env.max_steps):
        action = env.action_space.sample()  
        observation, reward, done, info, _ = env.step(action)
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")

        if done:
            break

    env.render()
    env.close()
