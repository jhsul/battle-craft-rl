
import logging
import coloredlogs
from pprint import pprint
import gym
import minerl
import pickle

import numpy as np
from datetime import datetime


import matplotlib.pyplot as plt

# coloredlogs.install(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

import sys
from rewards import RewardsCalculator

sys.path.insert(0, "vpt")  # nopep8
from agent import MineRLAgent  # nopep8


n_trials = 200
rewards = []


env = gym.make("MineRLPunchCow-v0")
model = f"models/foundation-model-2x.model"

weights = f"weights/foundation-model-2x.weights"

agent_parameters = pickle.load(open(model, "rb"))
# pprint(agent_parameters)
policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

agent = MineRLAgent(env, policy_kwargs=policy_kwargs,
                    pi_head_kwargs=pi_head_kwargs)
agent.load_weights(weights)


rc = RewardsCalculator(
    damage_dealt=1
)

plt.ion()
fig = plt.figure()

n, bins, patches = plt.hist(
    [], bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])

for k in range(n_trials):
    start = datetime.now()
    print(f"🚩 Starting trial {k+1}/{n_trials}")
    agent.reset()
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(obs)
        # action = env.action_space.noop()
        obs, reward, done, info = env.step(action)
        reward = rc.get_rewards(obs, verbose=True)
        total_reward += reward
        env.render()

    rc.clear()
    rewards.append(total_reward)
    print(
        f"🏁 Finished  with {total_reward} reward (time: {datetime.now() - start})\n")
    print(f"Current mean:   {np.mean(rewards)}")
    print(f"Current stddev: {np.std(rewards)}")
    print(f"(n={len(rewards)})")
    print()

    new_n, _ = np.histogram(rewards, bins=bins)
    for patch, new_value in zip(patches, new_n):
        patch.set_height(new_value)

    # Update the ylim to accommodate new data
    plt.ylim(0, np.max(new_n) * 1.1)

    # Redraw the canvas
    plt.draw()
    fig.canvas.flush_events()


env.close()
