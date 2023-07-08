from flask import Flask, jsonify, request
from flask_cors import CORS
import gym
import numpy as np
import pandas as pd
import yfinance as yf

from tradingEnv import TradingEnv

# TODO: Complete Functionality of App.py for Flask App 

app = Flask(__name__)
CORS(app)

env = TradingEnv()

@app.route('/reset', methods=['POST'])
def reset():
    env.reset()
    observation = env._get_observation()
    return jsonify(observation.tolist())

@app.route('/step', methods=['POST'])
def step():
    action = request.json['action']
    observation, reward, done, info = env.step(action)
    return jsonify({
        'observation': observation.tolist(),
        'reward': reward,
        'done': done,
        'info': info
    })

# Work in progress...
if __name__ == '__main__':
    app.run(debug=True)
