import pickle
import sys
import time
from typing import List
import gym
import torch as th
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

from torch.utils.tensorboard import SummaryWriter


from datetime import datetime
from tqdm import tqdm
from memory import Memory, MemoryDataset, AuxMemory
from util import to_torch_tensor, normalize, safe_reset, hard_reset, returns_and_advantages
from vectorized_minerl import *

sys.path.insert(0, "vpt")  # nopep8

from agent import MineRLAgent  # nopep8
from lib.tree_util import tree_map  # nopep8

# For debugging purposes
# th.autograd.set_detect_anomaly(True)
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
# device = th.device("mps")  # apple silicon

TRAIN_WHOLE_MODEL = False


class PhasicPolicyGradient:
    def __init__(
            self,
            env_name: str,
            model: str,
            weights: str,
            out_weights: str,
            save_every: int,


            # Hyperparameters
            num_rollouts: int,
            num_steps: int,  # the number of steps per rollout, T
            epochs: int,
            minibatch_size: int,
            lr: float,
            weight_decay: float,
            betas: tuple,
            beta_s: float,
            eps_clip: float,
            value_clip: float,
            value_loss_weight: float,
            gamma: float,
            lam: float,
            beta_klp: float,


            mem_buffer_size: int,
            sleep_cycles: int,
            beta_clone: float,

            # Optional: plot stuff
            plot: bool,
            num_envs: int

    ):
        model_path = f"models/{model}.model"
        weights_path = f"weights/{weights}.weights"
        self.out_weights_path = f"weights/{out_weights}.weights"
        self.training_name = f"ppo-{env_name}-{out_weights}-{int(time.time())}"

        self.env_name = env_name
        self.num_envs = num_envs
        self.envs = init_vec_envs(self.env_name, self.num_envs)

        self.save_every = save_every

        # Load hyperparameters unchanged
        self.num_rollouts = num_rollouts
        self.num_steps = num_steps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.beta_s = beta_s
        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.value_loss_weight = value_loss_weight
        self.gamma = gamma
        self.lam = lam
        self.beta_klp = beta_klp
        self.sleep_cycles = sleep_cycles
        self.beta_clone = beta_clone

        self.plot = plot

        self.mem_buffer_size = mem_buffer_size

        if self.plot:
            # These statistics are calculated live during the episode running (rollout)
            self.live_reward_history = []
            self.live_value_history = []
            self.live_gae_history = []

        # Load the agent parameters from the weight files
        agent_parameters = pickle.load(open(model_path, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

        # Make our agents

        self.agent = MineRLAgent(self.envs, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(weights_path)

        actor = self.agent.policy
        policy_params = actor.parameters()

        self.optim = th.optim.Adam(
            policy_params, lr=lr, betas=betas, weight_decay=weight_decay)

        self.scheduler = th.optim.lr_scheduler.LambdaLR(
            self.optim, lambda x: 1 - x / num_rollouts)

        # Create a SEPARATE VPT agent just for the critic
        self.critic = MineRLAgent(self.envs, policy_kwargs=policy_kwargs,
                                  pi_head_kwargs=pi_head_kwargs)
        self.critic.load_weights(weights_path)
        critic_params = list(self.critic.policy.value_head.parameters(
        )) + list(self.critic.policy.net.parameters())

        # separate optimizer for the critic
        self.critic_optim = th.optim.Adam(
            critic_params, lr=lr, betas=betas, weight_decay=weight_decay)

        self.scheduler_critic = th.optim.lr_scheduler.LambdaLR(
            self.critic_optim, lambda x: 1 - x / num_rollouts)

        # Internal buffer of the most recent episode memories
        # This will be a relatively large chunk of data
        # Potential memory issues / optimizations around here...
        self.memories: List[List[Memory]] = []
        self.aux_memories: List[List[AuxMemory]] = []

        # Initialize the ORIGINAL MODEL for a KL divergence term during the Policy Phase
        # We will use KL divergence between our policy predictions and the original policy
        # This is to ensure that we don't deviate too far from the original policy
        self.orig_agent = MineRLAgent(self.envs, policy_kwargs=policy_kwargs,
                                      pi_head_kwargs=pi_head_kwargs)
        self.orig_agent.load_weights(weights_path)

        # Initialize the hidden states of this agent
        self.orig_hidden_states = {}
        for i in range(self.num_envs):
            self.orig_hidden_states[i] = self.orig_agent.policy.initial_state(
                1)

        # Setup tensorboard logging
        self.tb_writer = SummaryWriter()

        # Used for indexing tensorboard plots
        self.num_wake_updates = 0
        self.num_sleep_updates = 0
        self.num_rollouts_so_far = 0  # name conflict with num_rollouts
        self.num_episodes_finished = 0

    def policy(self):
        '''
        Returns the policy network head, aux value head, and base
        '''

        return self.agent.policy.pi_head, self.agent.policy.value_head, self.agent.policy.net

    def value(self):
        '''
        Return the current value network  head and base
        '''
        return self.critic.policy.value_head, self.critic.policy.net

    def pi_and_v(self, agent_obs, policy_hidden_state, value_hidden_state, dummy_first, use_aux=False):
        """
        Returns the correct policy and value outputs
        """
        # Shorthand for networks
        policy, aux, policy_base = self.policy()
        value, value_base = self.value()

        (pi_h, aux_head), p_state_out = policy_base(
            agent_obs, policy_hidden_state, context={"first": dummy_first})
        (_, v_h), v_state_out = value_base(
            agent_obs, value_hidden_state, context={"first": dummy_first})

        if not use_aux:
            return policy(pi_h), value(v_h), p_state_out, v_state_out
        return policy(pi_h), value(v_h), aux(aux_head), p_state_out, v_state_out

    def init_plots(self):
        plt.ion()
        self.live_fig, self.live_ax = plt.subplots(1, 1, figsize=(6, 4))

        # Setup live plots
        self.live_ax.set_autoscale_on(True)
        self.live_ax.autoscale_view(True, True, True)
        self.live_ax.set_title("Episode Progress")
        self.live_ax.set_xlabel("steps")

        self.live_reward_plot, = self.live_ax.plot(
            [], [], color="red", label="Reward")
        self.live_value_plot, = self.live_ax.plot(
            [], [], color="blue", label="Value")
        self.live_gae_plot, = self.live_ax.plot(
            [], [], color="green", label="GAE")

        self.live_ax.legend(loc="upper right")

    def rollout(self, env, next_obs=None, next_done=False, next_policy_hidden_state=None, next_critic_hidden_state=None, hard_reset: bool = False):
        """
        Runs a rollout on the vectorized environments for `num_steps` timesteps and records the memories

        Returns `next_obs` and `next_done` for starting the next section of rollouts
        """
        start = datetime.now()

        # Temporary buffer to put the memories in before extending self.memories
        rollout_memories: List[Memory] = []

        # Only run this in the beginning
        if next_obs is None:
            # Initialize the hidden state vector
            next_policy_hidden_state = self.agent.policy.initial_state(1)
            next_critic_hidden_state = self.critic.policy.initial_state(1)

            if hard_reset:
                env.close()
                env = gym.make(self.env_name)
                next_obs = env.reset()
            else:
                next_obs = env.reset()

            # Need this after every call of env.reset()
            # Ideally we would use a gym wrapper
            # But this works for now and is a quick patch
            env._cum_reward = 0

            next_done = False

        # This is a dummy tensor of shape (batchsize, 1) which was used as a mask internally
        dummy_first = th.from_numpy(np.array((False,))).to(device)
        dummy_first = dummy_first.unsqueeze(1)

        # Keep track of this because why not
        rollout_reward = 0

        if self.plot:
            self.live_reward_history.clear()
            self.live_value_history.clear()
            self.live_gae_history.clear()

        for _ in range(self.num_steps):
            obs = next_obs
            done = next_done
            policy_hidden_state = next_policy_hidden_state
            critic_hidden_state = next_critic_hidden_state

            # We have to do some resetting...
            if done:

                next_obs = env.reset()
                env._cum_reward = 0
                policy_hidden_state = self.agent.policy.initial_state(1)
                critic_hidden_state = self.critic.policy.initial_state(1)

            # Preprocess image
            agent_obs = self.agent._env_obs_to_agent(obs)

            # Basically just adds a dimension to both camera and button tensors
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)

            with th.no_grad():
                pi_distribution, v_prediction, next_policy_hidden_state, next_critic_hidden_state \
                    = self.pi_and_v(agent_obs, policy_hidden_state, critic_hidden_state, dummy_first)

            action = self.agent.policy.pi_head.sample(
                pi_distribution, deterministic=False)

            # Get log probability of taking this action given pi
            action_log_prob = self.agent.policy.get_logprob_of_action(
                pi_distribution, action)

            # Process this so the env can accept it
            minerl_action = self.agent._agent_action_to_env(action)

            # Take action step in the environment
            next_obs, reward, next_done, info = env.step(minerl_action)
            env._cum_reward += reward  # Keep track of the episodic reawrd
            rollout_reward += reward

            if done:
                self.num_episodes_finished += 1
                self.tb_writer.add_scalar(
                    "Episodic Reward", env._cum_reward, self.num_episodes_finished)

            # Important! When we store a memory, we want the hidden state at the time of the observation as input! Not the step after
            # This is because we need to fully recreate the input when training the LSTM part of the network
            memory = Memory(agent_obs, 0, 0, 0, action, action_log_prob,
                            reward, 0, next_done, v_prediction)

            rollout_memories.append(memory)

            # Finally, render the environment to the screen
            # Comment this out if you are boring
            env.render()

            if self.plot:
                with torch.no_grad():
                    # Calculate the GAE up to this point
                    v_preds = list(
                        map(lambda mem: mem.advantage, rollout_memories))
                    rewards = list(
                        map(lambda mem: mem.returns, rollout_memories))
                    masks = list(
                        map(lambda mem: 1 - float(mem.done), rollout_memories))

                    agent_obs = self.agent._env_obs_to_agent(obs)
                    agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)
                    pi_distribution, v_prediction, next_policy_hidden_state, next_critic_hidden_state \
                        = self.pi_and_v(agent_obs, policy_hidden_state, critic_hidden_state, dummy_first)
                    returns = returns_and_advantages(
                        rewards, v_preds, masks, self.gamma, self.lam, v_prediction)

                    # Update data
                    self.live_reward_history.append(reward)
                    self.live_value_history.append(v_prediction.item())
                    self.live_gae_history = returns

                    # Update the plots
                    self.live_reward_plot.set_ydata(self.live_reward_history)
                    self.live_reward_plot.set_xdata(
                        range(len(self.live_reward_history)))

                    self.live_value_plot.set_ydata(self.live_value_history)
                    self.live_value_plot.set_xdata(
                        range(len(self.live_value_history)))

                    self.live_gae_plot.set_ydata(self.live_gae_history)
                    self.live_gae_plot.set_xdata(
                        range(len(self.live_gae_history)))

                    self.live_ax.relim()
                    self.live_ax.autoscale_view(True, True, True)

                    # Actually draw everything
                    self.live_fig.canvas.draw()
                    self.live_fig.canvas.flush_events()

        # Calculate the generalized advantage estimate
        v_preds = list(map(lambda mem: mem.advantage, rollout_memories))
        rewards = list(map(lambda mem: mem.returns, rollout_memories))
        masks = list(map(lambda mem: 1 - float(mem.done), rollout_memories))

        with torch.no_grad():
            agent_obs = self.agent._env_obs_to_agent(obs)
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)
            pi_distribution, v_prediction, next_policy_hidden_state, next_critic_hidden_state \
                = self.pi_and_v(agent_obs, policy_hidden_state, critic_hidden_state, dummy_first)
            returns = returns_and_advantages(
                rewards, v_preds, masks, self.gamma, self.lam, v_prediction)

        # Make changes to the memories for this episode before adding them to main buffer
        for i in range(len(rollout_memories)):
            # Replace raw reward with the GAE
            rollout_memories[i].returns = returns[i]

            # Remember the total reward for this episode
            # TODO this is broken for PPG!
            rollout_memories[i].total_reward = rollout_reward

        # Update internal memory buffer
        self.memories.append(rollout_memories)

        end = datetime.now()
        print(
            f"✅ Rollout finished (duration: {end - start} | memories: {len(rollout_memories)} | rollout reward: {rollout_reward})")

        return next_obs, next_done, next_policy_hidden_state, next_critic_hidden_state

    def learn_ppo_phase(self, policy_hidden_states, critic_hidden_states):

        # For simplicity
        assert self.num_envs % self.minibatch_size == 0
        memories_arr = np.array(self.memories)

        for _ in tqdm(range(self.epochs), desc="🧠 Policy Epochs"):

            # For training the recurrent model, we do batches of *sequences*
            # that is why this is `self.num_envs` and not `self.num_envs * self.num_timesteps`
            inds = np.arange(self.num_envs)
            np.random.shuffle(inds)
            minibatches = np.reshape(
                inds, (self.minibatch_size, self.num_envs // self.minibatch_size))

            # Note: These are SEQUENCES, NOT individual STPES
            for mb_inds in minibatches:
                rollouts = memories_arr[mb_inds]

                # All of the before the next for loop is to get the correct initial state to
                # start rerolling out the memory, it gave me mean warnings when I didn't
                # do it :(
                if policy_hidden_states[0] is None:
                    # Initialize the hidden state vector
                    policy_hidden_states = [
                        self.agent.policy.initial_state(1) for _ in mb_inds]
                    critic_hidden_states = [
                        self.critic.policy.initial_state(1) for _ in mb_inds]

                # Get the initial states for the current minibatches
                # use np.empty for making this because np is dumb
                mb_policy_states = np.empty(self.num_envs, dtype=object)
                mb_critic_states = np.empty(self.num_envs, dtype=object)

                mb_policy_states[:] = policy_hidden_states
                mb_critic_states[:] = critic_hidden_states

                mb_policy_states = mb_policy_states[mb_inds]
                mb_critic_states = mb_critic_states[mb_inds]

                policy_losses = []
                value_losses = []

                for rollout, policy_hidden_state, critic_hidden_state, i in \
                        zip(rollouts, mb_policy_states, mb_critic_states, mb_inds):

                    # Want these vectorized for later calculations
                    rewards = torch.tensor([mem.reward for mem in rollout])
                    old_action_log_probs = torch.tensor(
                        [mem.action_log_prob for mem in rollout])

                    # reroll out the memories using the same initial hidden state
                    # I think rerolling out is important for correct gradients for the
                    # recurrent model? that is basically speculation, though
                    log_probs = []
                    new_values = []
                    entropy = []
                    orig_pi_dists = []
                    pi_dists = []
                    dummy_first = th.from_numpy(
                        np.array((False,))).unsqueeze(1)
                    aux_mems = []
                    orig_hidden_state = self.orig_hidden_states[i]
                    for memory in rollout:
                        # Save the auxillary memories here, too
                        aux_mem = AuxMemory(
                            memory.agent_obs, memory.reward, memory.done)
                        aux_mems.append(aux_mem)

                        # Now start the actual rolling out
                        agent_obs = memory.agent_obs
                        pi_distribution, v_prediction, policy_hidden_state, critic_hidden_state \
                            = self.pi_and_v(agent_obs, policy_hidden_state, critic_hidden_state, dummy_first)

                        # Get the log prob of the RECORDED action based on the NEW pi_dist
                        action_log_prob = self.agent.policy.get_logprob_of_action(
                            pi_distribution, memory.action)

                        # Entropy of NEW pi_dist
                        e = self.agent.policy.pi_head.entropy(pi_distribution)

                        # Get the distributions for the ORIGINAL MODEL
                        # TODO save model hidden state?
                        with th.no_grad():
                            (pi_h, _), orig_hidden_state = self.orig_agent.policy.net(
                                agent_obs, orig_hidden_state, context={"first": dummy_first})

                            orig_pi_distribution = self.orig_agent.policy.pi_head(
                                pi_h)

                        log_probs.append(action_log_prob)
                        new_values.append(v_prediction)
                        entropy.append(e)
                        orig_pi_dists.append(orig_pi_distribution)
                        pi_dists.append(pi_distribution)

                        # If we are moving on to a new episode, reset the state
                        if memory.done:
                            policy_hidden_state = self.agent.policy.initial_state(
                                1)
                            critic_hidden_state = self.critic.policy.initial_state(
                                1)
                            orig_hidden_state = self.orig_agent.policy.initial_state(
                                1)

                    self.aux_memories.append(aux_mems)

                    action_log_probs = torch.cat(log_probs).squeeze()
                    v_prediction = torch.cat(new_values).squeeze()
                    entropy = torch.cat(entropy).squeeze()

                    orig_pi_dists = {'camera': torch.cat([p_dist['camera'] for p_dist in orig_pi_dists]),
                                     'buttons': torch.cat([p_dist['buttons'] for p_dist in orig_pi_dists]), }
                    pi_dists = {'camera': torch.cat([p_dist['camera'] for p_dist in pi_dists]),
                                'buttons': torch.cat([p_dist['buttons'] for p_dist in pi_dists]), }

                    # The returns are stored in the `reward` field in memory, for some reason
                    returns = normalize(rewards)

                    # Calculate the explained variance, to see how accurate the GAE really is...
                    explained_variance = 1 - \
                        th.sub(returns, v_prediction).var() / returns.var()

                    # Calculate clipped surrogate objective loss
                    ratios = (action_log_probs -
                            old_action_log_probs).exp().to(device)
                    advantages = returns - v_prediction.detach().to(device)
                    surr1 = ratios * advantages
                    surr2 = ratios.clamp(
                        1 - self.eps_clip, 1 + self.eps_clip) * advantages
                    kl_term = self.agent.policy.pi_head.kl_divergence(
                        orig_pi_dists, pi_dists)
                    policy_loss = - \
                        th.min(surr1, surr2) - self.beta_s * \
                        entropy - self.beta_klp * kl_term

                    # Calculate unclipped value loss
                    value_loss = 0.5 * (v_prediction.squeeze() - returns) ** 2

                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)

                # SAVE THE OUTPUT OF THE ORIG MODEL FOR NEXT PHASE
                # as dict for ease of access in minibatch...
                self.orig_hidden_states[i] = orig_hidden_state

                # accumulate the loss from the minibatches
                # even though we consider minibatches of sequences, we are still taking the
                # mean of the loss calculated for each STEP, so we are not losing out on much
                policy_loss = torch.cat(policy_losses).mean()
                value_loss = torch.cat(value_losses).mean()

                # Backprop for policy
                self.optim.zero_grad()
                policy_loss.backward()
                self.optim.step()

                # Backprop for critic
                self.critic_optim.zero_grad()
                value_loss.backward()
                self.critic_optim.step()

                # Update tensorboard with metrics
                self.tb_writer.add_scalar(
                    "Loss/Wake/Policy", policy_loss.mean().item(), self.num_wake_updates)
                self.tb_writer.add_scalar(
                    "Loss/Sleep/Value (Critic)", value_loss.item(), self.num_wake_updates)

                self.tb_writer.add_scalar(
                    "Stats/Entropy", entropy.mean().item(), self.num_wake_updates)
                self.tb_writer.add_scalar(
                    "Stats/KL Divergence wrt Pretrained", kl_term.mean().item(), self.num_wake_updates)
                self.tb_writer.add_scalar(
                    "Stats/Explained Variance", explained_variance.item(), self.num_wake_updates)

                self.num_wake_updates += 1
        # Update learning rate
        # TODO how to handle this in the aux phase?
        self.scheduler.step()
        self.scheduler_critic.step()

    def calculate_policy_priors(self, policy_hidden_states, critic_hidden_states):
        '''
        Calculate the p_dist for the current policy for KL loss in the aux phase
        '''
        cached_p_dists_rollouts = []
        rollouts = self.aux_memories

        if policy_hidden_states[0] is None:
            # Initialize the hidden state vector
            policy_hidden_states = [self.agent.policy.initial_state(
                1) for _ in policy_hidden_states]
            critic_hidden_states = [self.critic.policy.initial_state(
                1) for _ in critic_hidden_states]

        for rollout, policy_hidden_state, critic_hidden_state in \
                zip(rollouts, policy_hidden_states, critic_hidden_states):

            # reroll out the memories using the same initial hidden state
            dummy_first = th.from_numpy(np.array((False,))).unsqueeze(1)
            pi_dists = []

            for memory in rollout:
                # Now start the actual rolling out
                agent_obs = memory.agent_obs
                pi_distribution, _, policy_hidden_state, critic_hidden_state \
                    = self.pi_and_v(agent_obs, policy_hidden_state, critic_hidden_state, dummy_first)

                pi_dists.append(pi_distribution)

                # If we are moving on to a new episode, reset the state
                if memory.done:
                    policy_hidden_state = self.agent.policy.initial_state(1)
                    critic_hidden_state = self.critic.policy.initial_state(1)

            cached_p_dists_rollouts.append(pi_dists)

        return cached_p_dists_rollouts

    def auxiliary_phase(self, policy_priors, policy_hidden_states, critic_hidden_states):
        '''
        Run the auxiliary training phase for the value and aux value functions
        '''
        aux_arr = np.array(self.aux_memories)
        # TODO plot for this phase too !!

        for _ in tqdm(range(self.sleep_cycles), desc="😴 Auxiliary Epochs"):

            # For training the recurrent model, we do batches of *sequences*
            # that is why this is `self.num_envs` and not `self.num_envs * self.num_timesteps`
            inds = np.arange(self.num_envs)
            np.random.shuffle(inds)
            minibatches = np.reshape(
                inds, (self.minibatch_size, self.num_envs // self.minibatch_size))

            # Note: These are SEQUENCES, NOT individual STPES
            for mb_inds in minibatches:
                rollouts = aux_arr[mb_inds]

                # All of the before the next for loop is to get the correct initial state to
                # start rerolling out the memory, it gave me mean warnings when I didn't
                # do it :(
                if policy_hidden_states[0] is None:
                    # Initialize the hidden state vector
                    policy_hidden_states = [
                        self.agent.policy.initial_state(1) for _ in mb_inds]
                    critic_hidden_states = [
                        self.critic.policy.initial_state(1) for _ in mb_inds]

                # Get the initial states for the current minibatches
                # use np.empty for making this because np is dumb
                mb_policy_states = np.empty(self.num_envs, dtype=object)
                mb_critic_states = np.empty(self.num_envs, dtype=object)

                mb_policy_states[:] = policy_hidden_states
                mb_critic_states[:] = critic_hidden_states

                mb_policy_states = mb_policy_states[mb_inds]
                mb_critic_states = mb_critic_states[mb_inds]

                joint_losses = []
                value_losses = []
                for rollout, priors, policy_hidden_state, critic_hidden_state in \
                        zip(rollouts, policy_priors, mb_policy_states, mb_critic_states):

                    # Want these vectorized for later calculations
                    v_targ = torch.tensor([mem.v_targ for mem in rollout])
                    # to vectorize the dists, you have to do it per element in the dict
                    p_dist_old = {'camera': torch.cat([p_dist['camera'].detach() for p_dist in priors]),
                                  'buttons': torch.cat([p_dist['buttons'].detach() for p_dist in priors]), }

                    # reroll out the memories using the same initial hidden state
                    new_values = []
                    new_pi_dists = []
                    new_aux_values = []
                    dummy_first = th.from_numpy(
                        np.array((False,))).unsqueeze(1)
                    for memory in rollout:

                        agent_obs = memory.agent_obs
                        pi_distribution, v_prediction, aux_pred, policy_hidden_state, critic_hidden_state \
                            = self.pi_and_v(agent_obs, policy_hidden_state, critic_hidden_state, dummy_first, use_aux=True)

                        new_pi_dists.append(pi_distribution)
                        new_values.append(v_prediction)
                        new_aux_values.append(aux_pred)

                        # If we are moving on to a new episode, reset the state
                        if memory.done:
                            policy_hidden_state = self.agent.policy.initial_state(
                                1)
                            critic_hidden_state = self.critic.policy.initial_state(
                                1)

                    pi_dists = {'camera': torch.cat([p_dist['camera'] for p_dist in new_pi_dists]),
                                'buttons': torch.cat([p_dist['buttons'] for p_dist in new_pi_dists]), }
                    aux_predication = torch.cat(new_aux_values).squeeze()
                    v_prediction = torch.cat(new_values).squeeze()

                    # Calculate joint loss
                    aux_loss = 0.5 * (aux_predication - v_targ.detach()) ** 2
                    kl_term = self.agent.policy.pi_head.kl_divergence(
                        p_dist_old, pi_dists)
                    joint_loss = aux_loss + self.beta_clone * kl_term
                    joint_losses.append(joint_loss)

                    # Calculate unclipped value loss
                    value_loss = 0.5 * (v_prediction - v_targ.detach()) ** 2
                    value_losses.append(value_loss)

                # accumulate the loss from the minibatches
                # even though we consider minibatches of sequences, we are still taking the
                # mean of the loss calculated for each STEP, so we are not losing out on much
                joint_loss = torch.cat(joint_losses).mean()
                value_loss = torch.cat(value_losses).mean()

                # optimize Ljoint wrt policy weights
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                # optimize Lvalue wrt value weights
                self.critic_optim.zero_grad()
                value_loss.backward()
                self.critic_optim.step()

                # Update tensorboard
                self.tb_writer.add_scalar(
                    "Loss/Sleep/Value (Joint)", joint_loss.item(), self.num_sleep_updates)

                self.tb_writer.add_scalar(
                    "Loss/Sleep/Value (Critic)", value_loss.item(), self.num_sleep_updates)

                self.num_sleep_updates += 1

    def run_train_loop(self):
        """
        Runs the basic PPG training loop
        """
        if self.plot:
            self.init_plots()

        obss = [None]*self.num_envs
        dones = [False]*self.num_envs
        policy_states = [None]*self.num_envs
        critic_states = [None]*self.num_envs

        for i in range(self.num_rollouts):

            if i % self.save_every == 0:
                state_dict = self.agent.policy.state_dict()
                th.save(state_dict, f'{self.out_weights_path}_{i}')
                print(
                    f"💾 Saved checkpoint weights to {self.out_weights_path}_{i}")

            print(
                f"🎬 Starting {self.env_name} rollout {i + 1}/{self.num_rollouts}")

            # Do a server restart every 10 rollouts
            # Note: This is not one-to-one with the episodes
            # At T = 100, this is every 5 episodes
            should_hard_reset = i % 10 == 0 and i != 0

            obss_buffer = []
            dones_buffer = []
            policy_states_buffer = []
            critic_states_buffer = []
            for env, next_obs, next_done, next_policy_hidden_state, next_critic_hidden_state \
                    in zip(self.envs, obss, dones, policy_states, critic_states):
                next_obs, next_done, next_policy_hidden_state, next_critic_hidden_state = self.rollout(
                    env, next_obs, next_done, next_policy_hidden_state, next_critic_hidden_state, hard_reset=should_hard_reset)
                obss_buffer.append(next_obs)
                dones_buffer.append(next_done)
                policy_states_buffer.append(next_policy_hidden_state)
                critic_states_buffer.append(next_critic_hidden_state)

            # Need to give initial states from this rollout to re-rollout in learning with LSTM model
            self.learn_ppo_phase(policy_states, critic_states)

            # we have aux memories now, clear this shit OUT
            self.memories.clear()

            # calculate policy priors
            with torch.no_grad():
                policy_priors = self.calculate_policy_priors(
                    policy_states, critic_states)

            self.auxiliary_phase(policy_priors, policy_states, critic_states)

            self.aux_memories.clear()

            # Update from buffers AFTER learning...
            obss = obss_buffer
            dones = dones_buffer
            policy_states = policy_states_buffer
            critic_states = critic_states_buffer


if __name__ == "__main__":

    ppg = PhasicPolicyGradient(
        env_name="MineRLPunchCowEz-v0",
        model="foundation-model-1x",
        weights="foundation-model-1x",
        out_weights="cow-deleter-ppo-2ent-1x",
        save_every=5,
        num_envs=4,
        num_rollouts=500,
        num_steps=50,
        epochs=1,
        minibatch_size=1,
        lr=2.5e-5,
        weight_decay=0,
        betas=(0.9, 0.999),
        beta_s=0,  # no entropy in fine tuning!
        eps_clip=0.2,
        value_clip=0.2,
        value_loss_weight=0.2,
        gamma=0.99,
        lam=0.95,
        beta_klp=1,
        sleep_cycles=2,
        beta_clone=1,
        mem_buffer_size=10000,
        plot=True,
    )

    ppg.run_train_loop()
