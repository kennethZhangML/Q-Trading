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

        self.observation_space = spaces.Box(low = 0, high = 1, shape = (self.num_stocks + 1, self.num_stocks))
        self.action_space = spaces.Box(low = -1, high = 1, shape = (self.num_stocks, self.num_stocks))

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
    
    def get_stock_weights_from_action(self, action):
        current_weights = np.array(self.portfolio_weights)
        adjustment = np.array(action)
        updated_weights = current_weights + adjustment
        updated_weights = np.clip(updated_weights, 0, 1)  # Clip weights between 0 and 1
        updated_weights /= np.sum(updated_weights)  # Normalize weights to ensure they sum up to 1
        return updated_weights.tolist()

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
            observation = np.concatenate([observation, action.flatten()])
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
        portfolio_changes = np.dot(price_changes, self.portfolio_weights)

        buy_threshold = 0.02
        sell_threshold = 0.01
        stop_loss = -0.03
        take_profit = 0.05

        reward = np.zeros(self.num_stocks)

        self.stop_loss_level = [None] * self.num_stocks
        self.take_profit_level = [None] * self.num_stocks

        for i in range(self.num_stocks):
            if action[i][0] == 0 and portfolio_changes[i] > buy_threshold:
                reward[i] = 1.0
                self.stop_loss_level[i] = current_prices[i] * (1 + stop_loss)
                self.take_profit_level[i] = current_prices[i] * (1 + take_profit)

            elif action[i][0] == 1 and portfolio_changes[i] < -sell_threshold:
                reward[i] = -1.0
                self.stop_loss_level[i] = None
                self.take_profit_level[i] = None

            elif self.stop_loss_level[i] is not None and current_prices[i] <= self.stop_loss_level[i]:
                reward[i] = -1.0
                self.stop_loss_level[i] = None
                self.take_profit_level[i] = None

            elif self.take_profit_level[i] is not None and current_prices[i] >= self.take_profit_level[i]:
                reward[i] = 1.0
                self.stop_loss_level[i] = None
                self.take_profit_level[i] = None

        return reward

    def render(self, mode = 'human'):
        if self.animation is None:
            self.fig, self.axes = plt.subplots(
                nrows=self.num_stocks + 1, figsize = (10, 6), sharex = True, facecolor = 'black'
            )

            self.fig.patch.set_facecolor('black')
            for ax in self.axes:
                ax.set_facecolor('black')
                ax.grid(color='white', linestyle = 'dotted')

            for i, ax in enumerate(self.axes[:-1]):
                ax.set_ylabel(f'{self.stock_tickers[i]} Price')
                ax.plot([], [], color='white')

            self.axes[-1].set_xlabel('Date')
            self.axes[-1].set_ylabel('Portfolio Value')
            self.axes[-1].plot([], [], color = 'white')

            self.animation = FuncAnimation(
                self.fig, self._animate, frames=len(self.prices), interval=100, blit=False
            )

            plt.tight_layout()
            plt.show()
        self.animation.event_source.start()


    def _animate(self, i):
        current_prices = self.prices.iloc[:i + 1]

        for ax, stock_prices, ticker in zip(self.axes[:-1], current_prices.values.T, self.stock_tickers):
            ax.lines[0].set_data(current_prices.index, stock_prices)
            ax.lines[0].set_color(get_stock_color(ticker))

            # Set neon color for axis labels
            ax.spines['left'].set_color('lime')
            ax.spines['bottom'].set_color('lime')
            ax.tick_params(axis = 'both', colors = 'lime')
            ax.yaxis.label.set_color('lime')
            ax.xaxis.label.set_color('lime')

        portfolio_values = np.dot(current_prices.values, self.portfolio_weights)
        self.axes[-1].lines[0].set_data(current_prices.index, portfolio_values)

        diffs = np.diff(portfolio_values)
        declining = np.where(diffs < 0)[0] + 1
        increasing = np.where(diffs >= 0)[0] + 1

        self.axes[-1].fill_between(current_prices.index, 0, portfolio_values, color = 'lightblue')
        self.axes[-1].fill_between(current_prices.index, 0, portfolio_values, where = portfolio_values >= 0, color = 'lightblue')
        self.axes[-1].fill_between(current_prices.index, 0, portfolio_values, where = portfolio_values < 0, color = 'red')

        date_formatter = mdates.DateFormatter('%Y-%m-%d')
        self.axes[-1].xaxis.set_major_formatter(date_formatter)

        # Place portfolio value number at the top
        self.axes[-1].yaxis.set_label_coords(-0.05, 1.02)

        # Set neon color for portfolio value axis label
        self.axes[-1].spines['left'].set_color('lime')
        self.axes[-1].spines['bottom'].set_color('lime')
        self.axes[-1].tick_params(axis = 'both', colors = 'lime')
        self.axes[-1].yaxis.label.set_color('lime')
        self.axes[-1].xaxis.label.set_color('lime')

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