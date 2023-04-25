
import logging
import coloredlogs
import fire
import gym
import minerl
import pickle
import pandas as pd
import time
from efficient_vpt import EfficientVPT
import torch

import numpy as np
from util import safe_reset

import matplotlib.pyplot as plt

# coloredlogs.install(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

import sys
from tqdm import tqdm

sys.path.insert(0, "vpt")  # nopep8

from lib.tree_util import tree_map  # nopep8


def main(
        env: str,
        model: str,
        weights: str,
        n: int = 100
):
    env_name = env
    model_path = f"models/{model}.model"
    weights_path = f"noted_weights/{weights}.weights"

    baseline_name = f"{env_name}&{model}&{weights}-2"

    print(f"Beginning baseline (n={n})")
    print("============================")
    print(f"Environment: {env_name}")
    print(f"Model:       {model}")
    print(f"Weights:     {weights}")

    env = gym.make(env_name)

    agent_parameters = pickle.load(open(model_path, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    vpt = EfficientVPT(env, policy_kwargs=policy_kwargs,
                        pi_head_kwargs=pi_head_kwargs, use_skip=True)

    vpt.load_weights(weights_path)

    plt.ion()
    fig = plt.figure()

    hist_n, bins, patches = plt.hist(
        [], bins=list(range(-200, 200, 10)), color="green", label="Episode Rewards")

    rewards = []
    killed = []
    damage = []

        # This is a dummy tensor of shape (batchsize, 1) which was used as a mask internally
    dummy_first = torch.from_numpy(np.array((False,)))
    dummy_first = dummy_first.unsqueeze(1)

    for eps in tqdm(range(n)):
        # # Hard reset every 10 episodes so we don't crash
        # if eps % 10 == 0 and eps > 0:
        #     env.close()
        #     env = gym.make(env_name)
        obs, env = safe_reset(env)
        hidden_state = vpt.initial_state(1)

        done = False
        total_reward = 0
        while not done:
            agent_obs = vpt._env_obs_to_agent(obs)
            # Basically just adds a dimension to both camera and button tensors
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)

            latent, hidden_state = vpt.run_vpt_base(agent_obs, hidden_state, dummy_first)
            pi_distribution = vpt.get_policy(latent)

            action = vpt.policy.pi_head.sample(
                pi_distribution, deterministic=False)
            
            action = vpt._agent_action_to_env(action)
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # env.render()

        rewards.append(total_reward)
        killed.append(obs["mob_kills"]["mob_kills"] > 1)
        damage.append(obs["damage_dealt"]["damage_dealt"])

        # Record data for the previously finished episode
        new_n, _ = np.histogram(rewards, bins=bins)
        for patch, new_value in zip(patches, new_n):
            patch.set_height(new_value)

        # Update the ylim to accommodate new data
        plt.ylim(0, np.max(new_n) * 1.1)

        # Redraw the canvas
        plt.draw()
        fig.canvas.flush_events()
        plt.savefig(f"data/{baseline_name}.png")

        df = pd.DataFrame(
            data={"damage": damage, "killed": killed, "reward": rewards})
        df.to_csv(f"data/{baseline_name}.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
