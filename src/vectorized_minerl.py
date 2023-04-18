import gym
import torch
import numpy as np
from util import safe_reset
from memory import Memory


def init_vec_envs(name: str, num_envs: int):
    """
    Return num_envs copies of the given mineRL env
    """
    return [gym.make(name) for _ in range(num_envs)]


def reset_vec_envs(envs):
    """
    Calls `.reset()` on all of the envs to initialize them.
    Does not do a hard reset.

    Returns the initial observations
    """
    return [env.reset() for env in envs]


def run_base_vec_envs(agent_obss, hidden_states, agent, device):
    """
    Runs the base (assume untrained) VPT model on the envs and returns the
    pi_heads, v_heads, pi_distributions, v_predications, and next_hidden_states of the network
    """

    pihs = []
    vhs = []

    pis = []
    vs = []
    states = []
    for agent_obs, hidden_state in zip(agent_obss, hidden_states):
        # This is a dummy tensor of shape (batchsize, 1) which was used as a mask internally
        dummy_first = torch.from_numpy(np.array((False,))).to(device)
        dummy_first = dummy_first.unsqueeze(1)
        with torch.no_grad():
            (pi_h, v_h), next_hidden_state = agent.policy.net(
                agent_obs, hidden_state, context={"first": dummy_first})

            pihs.append(pi_h)
            vhs.append(v_h)

            pi_distribution = agent.policy.pi_head(pi_h)
            pis.append(pi_distribution)
            v_prediction = agent.policy.value_head(v_h)
            vs.append(v_prediction)
            states.append(next_hidden_state)

    return pihs, vhs, pis, vs, states


def step_vec_envs(envs, actions):
    """
    Steps forwards the vectorized MineRL environments given the list of actions
    Returns lists of obs, actions, rewards for each env

    Will hard reset and envionment automatically if it terminates

    KwdArg: actions - a list of actions corresponding to each env
    """

    obss = []
    rewards = []
    dones = []

    for env, minerl_action in zip(envs, actions):
        obs, reward, done, info = env.step(minerl_action)

        # Handle resetting the env to contrinue rolling out
        if done:
            obs, env = safe_reset(env)

        obss.append(obs)
        rewards.append(rewards)
        dones.append(done)

    return obs, rewards, dones


def generate_vec_memories(agent_obss, hidden_states, pi_hs, v_hs, actions, action_log_probs,
                          rewards, dones, v_preds):
    return [Memory(agent_obs, hidden_state, pi_h, v_h, action, action_log_prob,
                   reward, 0, done, v_prediction) for agent_obs, hidden_state, pi_h, v_h,
            action, action_log_prob, reward, done, v_prediction in
            zip(agent_obss, hidden_states, pi_hs, v_hs,
                actions, action_log_probs, rewards, dones, v_preds)]
