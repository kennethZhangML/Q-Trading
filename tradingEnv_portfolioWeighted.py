import gym
from gym import spaces

import yfinance as yf
import numpy as np
import pandas as pd 
import warnings

import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
yf.pdr_override()

class TradingEnv(gym.Env):
    def __init__(self, stock_tickers, portfolio_weights, start_date, end_date):
        super(TradingEnv, self).__init__()

        self.stock_tickers = stock_tickers
        self.portfolio_weights = portfolio_weights
        self.num_stocks = len(stock_tickers)

        self.observation_space = spaces.Box(low = 0, high = 1, shape = (self.num_stocks + 1,))
        self.action_space = spaces.Discrete(3)

        self.current_step = 0
        self.start_date = start_date
        self.end_date = end_date
        self.prices = self._download_prices()
        self.max_steps = len(self.prices)
        self.stop_loss_level = None
        self.take_profit_level = None

        self.fig, self.ax = plt.subplots()
        self.line = self.ax.plot([], [])
        self.animation = None

    def _download_prices(self):
        stock_data = yf.download(self.stock_tickers, start = self.start_date, end = self.end_date)
        return stock_data['Close']

    def reset(self):
        self.current_step = 0
        self.stop_loss_level = None
        self.take_profit_level = None

        self.prices = self._download_prices()
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
            observation = np.zeros((self.num_stocks + 1,))
        else:
            stock_prices = self.prices.iloc[start_index:end_index].values
            portfolio_value = np.dot(stock_prices[-1], self.portfolio_weights)

            observation = np.concatenate([stock_prices.flatten(), [portfolio_value]])

        return observation

    def _calculate_reward(self, action):
        current_prices = self.prices.iloc[self.current_step]
        previous_prices = self.prices.iloc[self.current_step - 1]

        price_changes = current_prices - previous_prices
        portfolio_change = np.dot(price_changes, self.portfolio_weights)

        buy_threshold = 0.02 
        sell_threshold = 0.01  
        stop_loss = -0.03 
        take_profit = 0.05  

        if action == 0 and portfolio_change > buy_threshold:
            reward = 1.0 
            self.stop_loss_level = current_prices * (1 + stop_loss)
            self.take_profit_level = current_prices * (1 + take_profit)

        elif action == 1 and portfolio_change < -sell_threshold:
            reward = -1.0 
            self.stop_loss_level = None
            self.take_profit_level = None

        elif self.stop_loss_level is not None and np.any(current_prices <= self.stop_loss_level):
            reward = -1.0 
            self.stop_loss_level = None
            self.take_profit_level = None

        elif self.take_profit_level is not None and np.any(current_prices >= self.take_profit_level):
            reward = 1.0  
            self.stop_loss_level = None
            self.take_profit_level = None

        else:
            reward = 0.0  

        return reward

    def render(self, mode='human'):
        if self.animation is None:
            self.fig, self.axes = plt.subplots(nrows = self.num_stocks + 1, figsize=(10, 6), sharex=True)

            for i, ax in enumerate(self.axes[:-1]):
                ax.set_ylabel(f'{self.stock_tickers[i]} Price')
                ax.plot([], [])

            self.axes[-1].set_xlabel('Date')
            self.axes[-1].set_ylabel('Portfolio Value')
            self.axes[-1].plot([], [])

            self.animation = FuncAnimation(
                self.fig, self._animate, frames = len(self.prices), interval = 100, blit = False
            )

            plt.tight_layout()
            plt.show()

        self.animation.event_source.start()

    def _animate(self, i):
        current_prices = self.prices.iloc[:i+1]

        for ax, stock_prices, ticker in zip(self.axes[:-1], current_prices.values.T, self.stock_tickers):
            ax.lines[0].set_data(current_prices.index, stock_prices)
            ax.lines[0].set_color(get_stock_color(ticker))

        portfolio_values = np.dot(current_prices.values, self.portfolio_weights)
        self.axes[-1].lines[0].set_data(current_prices.index, portfolio_values)

        diffs = np.diff(portfolio_values)
        declining = np.where(diffs < 0)[0] + 1
        increasing = np.where(diffs >= 0)[0] + 1

        self.axes[-1].fill_between(current_prices.index, 0, portfolio_values, color = 'lightblue')
        self.axes[-1].fill_between(current_prices.index, 0, portfolio_values, where = portfolio_values >= 0, color='lightblue')
        self.axes[-1].fill_between(current_prices.index, 0, portfolio_values, where = portfolio_values < 0, color='red')

        date_formatter = mdates.DateFormatter('%Y-%m-%d')
        self.axes[-1].xaxis.set_major_formatter(date_formatter)

        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()

    def close(self):
        pass

# --------------------------------------------
# EXAMPLE: Stock Color Mapping Function
def get_stock_color(ticker):
    color_mapping = {
        'AAPL': 'red',
        'GOOGL': 'green',
        'MSFT': 'blue',
        'AMZN': 'orange'
    }
    return color_mapping.get(ticker, 'black')
# --------------------------------------------


if __name__ == "__main__":

    stock_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    portfolio_weights = [0.3, 0.2, 0.3, 0.2]

    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")

    env = TradingEnv(stock_tickers, portfolio_weights, start_date, end_date)
    observation = env.reset()

    for step in range(env.max_steps):
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)

        print(f"Step: {step}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        env.render()

        if done:
            break

    env.close()