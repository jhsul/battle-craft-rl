# Random functions / class definitions

import gym
import torch as th

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


def to_torch_tensor(t): return th.stack(t).to(device).detach()


def normalize(t, eps=1e-5):
    return (t - t.mean()) / (t.std() + eps)


def safe_reset(env):
    """
    Helper function that runs `env.reset()` until it succeeds
    """
    try:
        obs = env.reset()

    except Exception as e:
        print("Caught game crash! Trying again")
        env.close()
        env = gym.make(env.unwrapped.spec.id)
        return safe_reset(env)

    else:
        return obs, env


def hard_reset(env):
    env.close()
    env = gym.make(env.unwrapped.spec.id)
    obs = env.reset()
    return obs, env
