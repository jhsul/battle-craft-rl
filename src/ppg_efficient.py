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
import gc

from efficient_vpt import EfficientVPT
from datetime import datetime
from tqdm import tqdm
from memory import Memory, MemoryDataset, AuxMemory, IndexedMemoryDataset
from util import to_torch_tensor, normalize, safe_reset, hard_reset, calculate_gae
from vectorized_minerl import *

sys.path.insert(0, "vpt")  # nopep8

from agent import MineRLAgent  # nopep8
from lib.tree_util import tree_map  # nopep8

# For debugging purposes
# th.autograd.set_detect_anomaly(True)
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
# device = th.device("mps")  # apple silicon

TRAIN_WHOLE_MODEL = False



class EfficientPhasicPolicyGradient:
    def __init__(
            self,
            env_name: str,
            model: str,
            weights: str,
            out_weights: str,
            save_every: int,


            # Hyperparameters
            num_rollouts: int,
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

        # no more vectorized envs :( just rollout the whole thing instead
        # The tradeoff for needing to create many servers is NOT worth it!
        self.env = gym.make(env_name)

        self.save_every = save_every

        # Load hyperparameters unchanged
        self.num_rollouts = num_rollouts
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
            self.pi_loss_history = []
            self.v_loss_history = []
            self.total_loss_history = []

            self.entropy_history = []
            self.expl_var_history = []

            self.surr1_history = []
            self.surr2_history = []

            self.reward_history = []

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

        self.model = EfficientVPT(self.env, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.model.load_vpt_weights(weights_path)

        self.policy_optim = th.optim.Adam(self.model.policy_parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        
        self.scheduler = th.optim.lr_scheduler.LambdaLR(
            self.policy_optim, lambda x: 1 - x / num_rollouts)
        
        # separate optimizer for the critic
        self.critic_optim = th.optim.Adam(self.model.value_parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        self.scheduler_critic = th.optim.lr_scheduler.LambdaLR(
        self.critic_optim, lambda x: 1 - x / num_rollouts)

        # Internal buffer of the most recent episode memories
        # This will be a relatively large chunk of data
        # Potential memory issues / optimizations around here...
        self.memories: List[Memory] = []


        # # Initialize the ORIGINAL MODEL for a KL divergence term during the Policy Phase
        # # We will use KL divergence between our policy predictions and the original policy
        # # This is to ensure that we don't deviate too far from the original policy
        # self.orig_agent = MineRLAgent(self.envs, policy_kwargs=policy_kwargs,
        #                               pi_head_kwargs=pi_head_kwargs)
        # self.orig_agent.load_weights(weights_path)
        
        # # initialize the hidden states of this agent
        # self.orig_hidden_states = {}
        # for i in range(self.num_envs):
        #     self.orig_hidden_states[i] = self.orig_agent.policy.initial_state(1)



    def pi_and_v(self, agent_obs, hidden_state, dummy_first, use_aux=False):
        """
        Returns the correct policy and value outputs
        """
        latent, hidden_state_out = self.model.run_vpt_base(agent_obs, hidden_state, dummy_first)

        pi_dist = self.model.get_policy(latent)
        v_pred = self.model.get_real_value(latent)
        
        if not use_aux:
            return pi_dist, v_pred, hidden_state_out, latent
        
        aux = self.model.get_aux_value(latent)

        return pi_dist, v_pred, aux, hidden_state_out, latent


    def init_plots(self):
        plt.ion()
        self.main_fig, self.main_ax = plt.subplots(2, 3, figsize=(12, 8))
        self.live_fig, self.live_ax = plt.subplots(1, 1, figsize=(6, 4))

        # Set up policy loss plot
        self.main_ax[0, 0].set_autoscale_on(True)
        self.main_ax[0, 0].autoscale_view(True, True, True)

        self.main_ax[0, 0].set_title("Policy Loss")

        self.pi_loss_plot, = self.main_ax[0, 0].plot(
            [], [], color="blue")

        # Setup value loss plot
        self.main_ax[0, 1].set_autoscale_on(True)
        self.main_ax[0, 1].autoscale_view(True, True, True)

        self.main_ax[0, 1].set_title("Value Loss")

        self.v_loss_plot, = self.main_ax[0, 1].plot(
            [], [], color="orange")

        # Set up total loss plot
        self.main_ax[0, 2].set_autoscale_on(True)
        self.main_ax[0, 2].autoscale_view(True, True, True)

        self.main_ax[0, 2].set_title("Total Loss")

        self.total_loss_plot, = self.main_ax[0, 2].plot(
            [], [], color="purple"
        )

        # Setup entropy plot
        self.main_ax[1, 0].set_autoscale_on(True)
        self.main_ax[1, 0].autoscale_view(True, True, True)
        self.main_ax[1, 0].set_title("Entropy")

        self.entropy_plot, = self.main_ax[1, 0].plot([], [], color="green")

        # Setup explained variance plot
        self.main_ax[1, 1].set_autoscale_on(True)
        self.main_ax[1, 1].autoscale_view(True, True, True)
        self.main_ax[1, 1].set_title("Explained Vaiance")

        self.expl_var_plot, = self.main_ax[1, 1].plot([], [], color="grey")

        # Setup reward plot
        self.main_ax[1, 2].set_autoscale_on(True)
        self.main_ax[1, 2].autoscale_view(True, True, True)
        self.main_ax[1, 2].set_title("Reward per Rollout Phase")

        self.reward_plot,  = self.main_ax[1, 2].plot([], [], color="red")

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

    def rollout(self, hard_reset: bool = False):
        """
        Runs a rollout on the environment and records the memories
        """
        start = datetime.now()

        # Temporary buffer to put the memories in before extending self.memories
        rollout_memories: List[Memory] = []

        if hard_reset:
            self.env.close()
            self.env = gym.make(self.env_name)
            next_obs = self.env.reset()
        else:
            next_obs = self.env.reset()

        done = False

        # This is a dummy tensor of shape (batchsize, 1) which was used as a mask internally
        dummy_first = th.from_numpy(np.array((False,))).to(device)
        dummy_first = dummy_first.unsqueeze(1)

        hidden_state = self.model.initial_state(1)

        # This is not really used in training
        # More just for us to estimate the success of an episode
        episode_reward = 0

        if self.plot:
            self.live_reward_history.clear()
            self.live_value_history.clear()
            self.live_gae_history.clear()

        while not done:
            obs = next_obs

            # Preprocess image
            agent_obs = self.model._env_obs_to_agent(obs)

            # Basically just adds a dimension to both camera and button tensors
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)

            with th.no_grad():
                pi_distribution, v_prediction, hidden_state, latent \
                    = self.pi_and_v(agent_obs, hidden_state, dummy_first)
                

            action = self.model.policy.pi_head.sample(
                pi_distribution, deterministic=False)

            # Get log probability of taking this action given pi
            action_log_prob = self.model.policy.get_logprob_of_action(
                pi_distribution, action)

            # Process this so the env can accept it
            minerl_action = self.model._agent_action_to_env(action)

            # Take action step in the environment
            next_obs, reward, done, _ = self.env.step(minerl_action)
            episode_reward += reward

            # Important! When we store a memory, we want the hidden state at the time of the observation as input! Not the step after
            # This is because we need to fully recreate the input when training the LSTM part of the network
            memory = Memory(0, 0, latent, 0, action, action_log_prob,
                            reward, 0, done, v_prediction)

            rollout_memories.append(memory)

            # Finally, render the environment to the screen
            # Comment this out if you are boring
            self.env.render()

            if self.plot:
                with torch.no_grad():
                    # Calculate the GAE up to this point
                    v_preds = list(map(lambda mem: mem.value, rollout_memories))
                    rewards = list(map(lambda mem: mem.reward, rollout_memories))
                    masks = list(
                        map(lambda mem: 1 - float(mem.done), rollout_memories))

                    agent_obs = self.model._env_obs_to_agent(obs)
                    agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)
                    _, v_prediction, _, _ \
                        = self.pi_and_v(agent_obs, hidden_state, dummy_first)
                    returns = calculate_gae(
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
        v_preds = list(map(lambda mem: mem.value, rollout_memories))
        rewards = list(map(lambda mem: mem.reward, rollout_memories))
        masks = list(map(lambda mem: 1 - float(mem.done), rollout_memories))

        with torch.no_grad():
            agent_obs = self.model._env_obs_to_agent(obs)
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)
            _, v_prediction, _, _ \
                = self.pi_and_v(agent_obs, hidden_state, dummy_first)
            returns = calculate_gae(rewards, v_preds, masks, self.gamma, self.lam, v_prediction)

        # Make changes to the memories for this episode before adding them to main buffer
        for i in range(len(rollout_memories)):
            # Replace raw reward with the GAE
            rollout_memories[i].reward = returns[i]

            # Remember the total reward for this episode
            rollout_memories[i].total_reward = episode_reward # TODO this is broken for PPG!

        # Update internal memory buffer
        self.memories.extend(rollout_memories)

        if self.plot:
            # Update the reward plot
            self.reward_history.append(episode_reward)
            self.reward_plot.set_ydata(self.reward_history)
            self.reward_plot.set_xdata(range(len(self.reward_history)))

            self.main_ax[1, 2].relim()
            self.main_ax[1, 2].autoscale_view(True, True, True)

            self.main_fig.canvas.draw()
            self.main_fig.canvas.flush_events()

        end = datetime.now()
        print(
            f"✅ Rollout finished (duration: {end - start} | memories: {len(rollout_memories)} | total reward: {episode_reward})")

    def learn_ppo_phase(self):

        # Create dataloader from the memory buffer
        data = MemoryDataset(self.memories)

        dl = DataLoader(data, batch_size=self.minibatch_size, shuffle=True)

        for _ in tqdm(range(self.epochs), desc="🧠 Policy Epochs"):

            # Note: These are batches, not individual samples
            for _, _, latent, _, actions, old_action_log_probs, rewards, _, _, _ in dl:
                
                v_prediction = self.model.get_real_value(latent)
                pi_distribution = self.model.get_policy(latent)


                action_log_probs = self.model.policy.get_logprob_of_action(
                    pi_distribution, actions)

                # Calculate entropy
                # TODO just remove this?
                entropy = self.model.policy.pi_head.entropy(
                    pi_distribution).to(device)

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
                # kl_term = self.agent.policy.pi_head.kl_divergence(orig_pi_dists, pi_dists)
                policy_loss = - th.min(surr1, surr2) - self.beta_s * entropy # - self.beta_klp * kl_term

                # Calculate unclipped value loss
                value_loss = 0.5 * (v_prediction.squeeze() - returns) ** 2

                # Backprop for policy
                self.policy_optim.zero_grad()
                policy_loss.mean().backward()
                self.policy_optim.step()

                # Backprop for critic
                self.critic_optim.zero_grad()
                value_loss.mean().backward()
                self.critic_optim.step()

                if self.plot:
                    self.pi_loss_history.append(policy_loss.mean().item())
                    self.v_loss_history.append(value_loss.mean().item())

                    self.expl_var_history.append(explained_variance.item())

                    self.entropy_history.append(entropy.mean().item())

            # Update plot at the end of every epoch
            if self.plot:
                # Update policy loss plot
                self.pi_loss_plot.set_ydata(self.pi_loss_history)
                self.pi_loss_plot.set_xdata(
                    range(len(self.pi_loss_history)))
                self.main_ax[0, 0].relim()
                self.main_ax[0, 0].autoscale_view(True, True, True)

                # Update value loss plot
                self.v_loss_plot.set_ydata(self.v_loss_history)
                self.v_loss_plot.set_xdata(
                    range(len(self.v_loss_history)))
                self.main_ax[0, 1].relim()
                self.main_ax[0, 1].autoscale_view(True, True, True)

                # Update total loss plot
                self.total_loss_plot.set_ydata(self.total_loss_history)
                self.total_loss_plot.set_xdata(
                    range(len(self.total_loss_history)))
                self.main_ax[0, 2].relim()
                self.main_ax[0, 2].autoscale_view(True, True, True)

                # Update the entropy plot
                self.entropy_plot.set_ydata(self.entropy_history)
                self.entropy_plot.set_xdata(range(len(self.entropy_history)))
                self.main_ax[1, 0].relim()
                self.main_ax[1, 0].autoscale_view(True, True, True)

                # Update the explained variance plot
                self.expl_var_plot.set_ydata(self.expl_var_history)
                self.expl_var_plot.set_xdata(
                    range(len(self.expl_var_history)))

                self.main_ax[1, 1].relim()
                self.main_ax[1, 1].autoscale_view(True, True, True)

                # Actually draw everything
                self.main_fig.canvas.draw()
                self.main_fig.canvas.flush_events()

        # Update learning rate
        # TODO how to handle this in the aux phase?
        self.scheduler.step()
        self.scheduler_critic.step()

    def calculate_policy_priors(self):
        '''
        Calculate the p_dist for the current policy for KL loss in the aux phase

        A bit of a cheat, but oh well
        '''

        # Get a dataloader for ALL memories to easily vectorize them
        # as long as there aren't too many memories, this should be fine
        data = MemoryDataset(self.memories)
        dl = DataLoader(data, batch_size=len(self.memories), shuffle=False)

        # This for should only iterate once
        for _, _, latent, _, _, _, _, _, _, _ in dl:
            pi_distribution = self.model.get_policy(latent)

        return pi_distribution
                
    def auxiliary_phase(self, policy_priors):
        '''
        Run the auxiliary training phase for the value and aux value functions
        '''
        # Create dataloader from the memory buffer
        data = IndexedMemoryDataset(self.memories)

        dl = DataLoader(data, batch_size=self.minibatch_size, shuffle=True)

        for _ in tqdm(range(self.sleep_cycles), desc="😴 Auxiliary Epochs"):

            # Note: These are batches, not individual samples
            for _, _, latent, _, _, _, rewards, _, _, _, idx in dl:

                p_dist_old = {'camera': policy_priors['camera'].index_select(0,idx),
                              'buttons': policy_priors['buttons'].index_select(0,idx)}
                v_prediction = self.model.get_real_value(latent)
                pi_dists = self.model.get_policy(latent)
                aux_prediction = self.model.get_aux_value(latent)

                # The returns are stored in the `reward` field in memory, for some reason
                # TODO do this on the dataset level instead
                v_targ = normalize(rewards)
 
                # Calculate joint loss
                aux_loss = 0.5 * (aux_prediction - v_targ.detach()) ** 2
                kl_term = self.model.policy.pi_head.kl_divergence(
                    p_dist_old, pi_dists)
                joint_loss = aux_loss + self.beta_clone * kl_term

                # Calculate unclipped value loss
                value_loss = 0.5 * (v_prediction - v_targ.detach()) ** 2

                # optimize Ljoint wrt policy weights
                self.policy_optim.zero_grad()
                joint_loss.mean().backward()
                self.policy_optim.step()

                # optimize Lvalue wrt value weights
                self.critic_optim.zero_grad()
                value_loss.mean().backward()
                self.critic_optim.step()

    def run_train_loop(self):
        """
        Runs the basic PPG training loop
        """
        if self.plot:
            self.init_plots()

        for i in range(self.num_rollouts):

            if i % self.save_every == 0:
                state_dict = self.model.state_dict()
                th.save(state_dict, f'{self.out_weights_path}_{i}')

                data_path = f"out/{self.training_name}.csv"
                df = pd.DataFrame(
                    data={
                        "pi_loss": self.pi_loss_history,
                        "v_loss": self.v_loss_history,
                        "entropy": self.entropy_history,
                        "expl_var": self.expl_var_history
                    })
                df.to_csv(data_path, index=False)

                fig_path = f"out/{self.training_name}.png"
                self.main_fig.savefig(fig_path)
                print(f"💾 Saved checkpoint data")
                print(f"   - {self.out_weights_path}")
                print(f"   - {data_path}")
                print(f"   - {fig_path}")

            print(
                f"🎬 Starting {self.env_name} rollout {i + 1}/{self.num_rollouts}")

            # Do a server restart every 10 rollouts
            # Note: This is not one-to-one with the episodes
            # At T = 100, this is every 5 episodes
            should_hard_reset = i % 10 == 0 and i != 0

            for _ in range(self.num_envs):
                self.rollout(hard_reset=should_hard_reset)

            # GAE is done in here, and memories=aux_memories
            self.learn_ppo_phase()

            # calculate policy priors
            with torch.no_grad():
                policy_priors = self.calculate_policy_priors()

            self.auxiliary_phase(policy_priors)

            self.memories.clear()


if __name__ == "__main__":

    ppg = EfficientPhasicPolicyGradient(
        env_name="MineRLPunchCowEz-v0",
        model="foundation-model-1x",
        weights="foundation-model-1x",
        out_weights="cow-deleter-ppg-eff-1x",
        save_every=5,
        num_envs=1,
        num_rollouts=500,
        epochs=1,
        minibatch_size=48,
        lr=2.5e-5,
        weight_decay=0,
        betas=(0.9, 0.999),
        beta_s=0, # no entropy in fine tuning!
        eps_clip=0.2,
        value_clip=0.2,
        value_loss_weight=0.2,
        gamma=0.99,
        lam=0.95,
        beta_klp = 1,
        sleep_cycles=2,
        beta_clone=1,
        mem_buffer_size=100000,
        plot=True,
    )

    ppg.run_train_loop()
