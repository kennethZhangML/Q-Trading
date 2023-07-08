# Q-Trading
An RL-agent based trading portfolio management program for day trading

# Section 1

## Trading Environment Readme

This code implements a custom trading environment using the Gym library. The environment allows for the training and testing of trading strategies using reinforcement learning algorithms.

### Environment Setup

To use this trading environment, you need to install the necessary dependencies:

- `gym`: The OpenAI Gym library provides a framework for creating and interacting with reinforcement learning environments.
- `yfinance`: A Python library that provides a simple way to download financial market data from Yahoo Finance.

You can install the dependencies using `pip`: pip install gym yfinance

### Environment Overview

The `TradingEnv` class defines the trading environment and inherits from the `gym.Env` base class. It provides the following functionality:

- **Observation Space**: The observation space is defined as a `Box` space with a shape of `(10,)`, where each element represents the closing price of a specific time step.
- **Action Space**: The action space is defined as a `Discrete` space with three possible actions: 0 (buy), 1 (sell), and 2 (hold).
- **Reset**: The `reset()` method initializes the environment and returns the initial observation.
- **Step**: The `step(action)` method advances the environment by one time step based on the chosen action. It returns the next observation, reward, done flag, and additional info.
- **Reward Calculation**: The `_calculate_reward(action)` method determines the reward based on the chosen action and the price change between the current and previous time steps. The reward is influenced by buy and sell thresholds, stop loss, and take profit levels.
- **Rendering and Closing**: The `render()` and `close()` methods are provided but do not have any specific implementation.

### Usage

To use the `TradingEnv`, you can follow the example code provided in the `if __name__ == "__main__":` block. Here's an overview of the steps:

1. Register the custom environment with Gym using `gym.envs.register()`. Provide a unique ID and the entry point of the environment (`tradingEnv:TradingEnv`).
2. Create an instance of the environment using `gym.make()` and pass the registered ID.
3. Reset the environment to obtain the initial observation using `env.reset()`.
4. Iterate through the desired number of steps and perform actions using `env.step(action)`. The action can be chosen randomly (`env.action_space.sample()`) or determined by a learning agent.
5. Extract the returned observation, reward, done flag, and additional info from `env.step()`.
6. Continue the loop until the done flag is `True`.
7. Close the environment using `env.close()`.

Please note that this example code only demonstrates the basic usage of the trading environment. In a real-world scenario, you would typically integrate this environment with a reinforcement learning algorithm for training or testing trading strategies.

### Customization

You can customize the trading environment according to your specific requirements. Some possible modifications include:

- Modifying the observation space: You can change the shape or limits of the observation space to include different features or adjust the range of values.
- Modifying the action space: You can change the number and meaning of the available actions to fit your trading strategy.
- Adjusting reward calculations: You can modify the `_calculate_reward(action)` method to define different reward structures based on your trading objectives.
- Adding rendering capabilities: You can implement the `render()` method to visualize the environment or display relevant information during training or testing.

