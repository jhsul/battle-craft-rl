import pickle
import sys
from typing import List
import gym
import torch as th
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy

from datetime import datetime
from tqdm import tqdm
from rewards import RewardsCalculator
from memory import Memory, MemoryDataset
from util import to_torch_tensor, normalize, safe_reset, hard_reset, calculate_gae

sys.path.insert(0, "vpt")  # nopep8

from agent import MineRLAgent  # nopep8
from lib.tree_util import tree_map  # nopep8

# For debugging purposes
# th.autograd.set_detect_anomaly(True)
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
# device = th.device("mps")  # apple silicon

TRAIN_WHOLE_MODEL = True


class ProximalPolicyOptimizer:
    def __init__(
            self,
            env_name: str,
            model_path: str,
            weights_path: str,

            # A custom reward function to be used
            rc: RewardsCalculator,

            # Hyperparameters
            ppo_iterations: int,
            episodes: int,
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

            mem_buffer_size: int,
            # tau: float = 0.95,

            # Optional: plot stuff
            plot: bool,



    ):
        self.env_name = env_name
        self.env = gym.make(env_name)

        # Load the reward calculator
        self.rc = rc

        # Load hyperparameters unchanged
        self.ppo_iterations = ppo_iterations
        self.episodes = episodes
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
        # self.tau = tau

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

        self.agent = MineRLAgent(self.env, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(weights_path)

        policy = self.agent.policy

        if TRAIN_WHOLE_MODEL:
            self.trainable_parameters = list(policy.parameters())
        else:
            self.trainable_parameters = list(
                policy.pi_head.parameters()) + list(policy.value_head.parameters())

        self.optim = th.optim.Adam(
            self.trainable_parameters, lr=lr, betas=betas, weight_decay=weight_decay)

        # self.optim_pi = th.optim.Adam(
        #     self.agent.policy.pi_head.parameters(), lr=lr, betas=betas)

        # self.optim_v = th.optim.Adam(
        #     self.agent.policy.value_head.parameters(), lr=lr, betas=betas)

        # Internal buffer of the most recent episode memories
        # This will be a relatively large chunk of data
        # Potential memory issues / optimizations around here...
        self.memories: List[Memory] = []

    def run_episode(self, hard_reset: bool = False):
        """
        Runs a single episode and records the memories 
        """
        start = datetime.now()

        # Temporary buffer to put the memories in before extending self.memories
        episode_memories: List[Memory] = []

        # Initialize the hidden state vector
        # Note that batch size is 1 here because we are only running the agent
        # on a single episode
        hidden_state = self.agent.policy.initial_state(1)

        # Need to do a little bit of augmentation so the dataloader accepts the initial hidden state
        # This shouldn't affect anything; initial_state just uses None instead of empty tensors
        for i in range(len(hidden_state)):
            hidden_state[i] = list(hidden_state[i])
            hidden_state[i][0] = th.from_numpy(np.full(
                (1, 1, 128), False)).to(device)

        # This is a dummy tensor of shape (batchsize, 1) which was used as a mask internally
        dummy_first = th.from_numpy(np.array((False,))).to(device)
        dummy_first = dummy_first.unsqueeze(1)

        # Start the episode with gym
        # obs = self.env.reset()
        if hard_reset:
            self.env.close()
            self.env = gym.make(self.env_name)
            obs = self.env.reset()
        else:
            obs = self.env.reset()
        done = False

        # This is not really used in training
        # More just for us to estimate the success of an episode
        episode_reward = 0

        if self.plot:
            self.live_reward_history.clear()
            self.live_value_history.clear()
            self.live_gae_history.clear()

        while not done:
            # Preprocess image
            agent_obs = self.agent._env_obs_to_agent(obs)

            # Basically just adds a dimension to both camera and button tensors
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)

            # print("Initial Hidden State:")
            # for i in range(len(hidden_state)):
            #     # print(hidden_state[i][0].shape)
            #     print(hidden_state[i][1][0].shape)
            #     print(hidden_state[i][1][1].shape)
            # Need to run this with no_grad or else the gradient descent will crash during training lol
            with th.no_grad():
                (pi_h, v_h), next_hidden_state = self.agent.policy.net(
                    agent_obs, hidden_state, context={"first": dummy_first})

                pi_distribution = self.agent.policy.pi_head(pi_h)
                v_prediction = self.agent.policy.value_head(v_h)

            # print("Next Hidden State:")
            # for i in range(len(next_hidden_state)):
            #     print(next_hidden_state[i][0].shape)
            #     print(next_hidden_state[i][1][0].shape)
            #     print(next_hidden_state[i][1][1].shape)
            # Get action sampled from policy distribution
            # If deterministic==True, this just uses argmax
            action = self.agent.policy.pi_head.sample(
                pi_distribution, deterministic=False)

            # print(action)

            # Get log probability of taking this action given pi
            action_log_prob = self.agent.policy.get_logprob_of_action(
                pi_distribution, action)

            # Process this so the env can accept it
            minerl_action = self.agent._agent_action_to_env(action)

            # Take action step in the environment
            obs, reward, done, info = self.env.step(minerl_action)

            # Immediately disregard the reward function from the environment
            reward = self.rc.get_rewards(obs)
            episode_reward += reward

            # Important! When we store a memory, we want the hidden state at the time of the observation as input! Not the step after
            # This is because we need to fully recreate the input when training the LSTM part of the network
            memory = Memory(agent_obs, hidden_state, pi_h, v_h, action, action_log_prob,
                            reward, 0, done, v_prediction)

            # Now, update the state to the new state
            hidden_state = next_hidden_state

            # print("Memory Hidden State: ", memory.hidden_state)

            episode_memories.append(memory)
            # Finally, render the environment to the screen
            # Comment this out if you are boring
            self.env.render()

            if self.plot:
                # Calculate the GAE up to this point
                v_preds = list(map(lambda mem: mem.value, episode_memories))
                rewards = list(map(lambda mem: mem.reward, episode_memories))
                masks = list(
                    map(lambda mem: 1 - float(mem.done), episode_memories))

                returns = calculate_gae(
                    rewards, v_preds, masks, self.gamma, self.lam)

                # Update data
                self.live_reward_history.append(reward)
                self.live_value_history.append(v_prediction.item())
                self.live_gae_history = returns

                # Update the plots
                self.live_reward_plot.set_ydata(self.live_reward_history)
                self.live_reward_plot.set_xdata(
                    np.arange(len(self.live_reward_history)))

                self.live_value_plot.set_ydata(self.live_value_history)
                self.live_value_plot.set_xdata(
                    np.arange(len(self.live_value_history)))

                self.live_gae_plot.set_ydata(self.live_gae_history)
                self.live_gae_plot.set_xdata(
                    np.arange(len(self.live_gae_history)))

                self.live_ax.relim()
                self.live_ax.autoscale_view(True, True, True)

                # Actually draw everything
                self.live_fig.canvas.draw()
                self.live_fig.canvas.flush_events()

        # Reset the reward calculator once we are done with the episode
        self.rc.clear()

        # Calculate the generalized advantage estimate
        # This used to be done during each minibatch; however, the GAE should be more accurate
        # if we calculate it over the entire episode... I think?
        v_preds = list(map(lambda mem: mem.value, episode_memories))
        rewards = list(map(lambda mem: mem.reward, episode_memories))
        masks = list(map(lambda mem: 1 - float(mem.done), episode_memories))

        returns = calculate_gae(rewards, v_preds, masks, self.gamma, self.lam)

        # for i in reversed(range(len(episode_memories))):

        #     # hacky but necessary since we don't have "next_state"
        #     v_next = v_preds[i + 1] if i != len(episode_memories) - 1 else 0

        #     delta = rewards[i] + self.gamma * \
        #         v_next * masks[i] - v_preds[i]
        #     gae = delta + self.gamma * self.lam * masks[i] * gae
        #     returns.insert(0, gae + v_preds[i])

        # Make changes to the memories for this episode before adding them to main buffer
        for i in range(len(episode_memories)):
            # Replace raw reward with the GAE
            episode_memories[i].reward = returns[i]

            # Remember the total reward for this episode
            episode_memories[i].total_reward = episode_reward

        # Update internal memory buffer
        self.memories.extend(episode_memories)

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
            f"✅ Episode finished (duration - {end - start} | memories - {len(episode_memories)} | total reward - {episode_reward})")

    def learn(self):

        # Create dataloader from the memory buffer
        data = MemoryDataset(self.memories)
        dl = DataLoader(data, batch_size=self.minibatch_size, shuffle=True)

        # Shorthand
        policy = self.agent.policy

        for _ in tqdm(range(self.epochs), desc="epochs"):

            # Note: These are batches, not individual samples
            for agent_obs, state, recorded_pi_h, recorded_v_h, actions, old_action_log_probs, rewards, total_rewards, dones, v_old in dl:
                batch_size = len(dones)
                # print(agent_obs["img"].shape)
                v_old = v_old.to(device)

                # DATALOADER WHY DO YOU LOVE ADDING DIMENSIONS?!!
                for i in range(len(state)):
                    state[i][0] = state[i][0].squeeze(1)
                    state[i][1][0] = state[i][1][0].squeeze(1)
                    state[i][1][1] = state[i][1][1].squeeze(1)
                agent_obs = tree_map(lambda x: x.squeeze(1), agent_obs)

                dummy_first = th.from_numpy(np.full(
                    (batch_size, 1), False)).to(device)

                # print(dummy_first.shape)

                if TRAIN_WHOLE_MODEL:
                    (pi_h, v_h), state_out = policy.net(
                        agent_obs, state, context={"first": dummy_first})

                else:
                    # Use the hidden state calculated at the time since the model shouldn't change
                    pi_h, v_h = recorded_pi_h, recorded_v_h

                pi_distribution = self.agent.policy.pi_head(pi_h)
                v_prediction = self.agent.policy.value_head(v_h).to(device)

                # Overwrite the rewards now
                # This line makes no sense because it is a holdover from when GAE was still calculated here
                # NORMALIZE THE GAE PER MINIBATCH
                returns = (rewards - th.mean(rewards)) / th.std(rewards)
                # returns = th.tensor(rewards).float().to(device)

                # Calculate the explained variance, to see how accurate the GAE really is...
                # print(rewards.shape)
                # print(v_prediction.shape)
                explained_variance = 1 - \
                    th.sub(returns, v_prediction).var() / returns.var()

                # print(rewards)
                # Get log probs
                action_log_probs = self.agent.policy.get_logprob_of_action(
                    pi_distribution, actions)

                entropy = self.agent.policy.pi_head.entropy(
                    pi_distribution).to(device)

                # Calculate clipped surrogate objective
                ratios = (action_log_probs -
                          old_action_log_probs).exp().to(device)
                advantages = normalize(
                    returns - v_prediction.detach().to(device))
                surr1 = ratios * advantages
                surr2 = ratios.clamp(
                    1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - th.min(surr1, surr2) - self.beta_s * entropy

                # Calculate clipped value loss
                value_clipped = (v_old +
                                 (v_prediction - v_old).clamp(-self.value_clip, self.value_clip)).to(device)

                value_loss_1 = (value_clipped.squeeze() - returns) ** 2
                value_loss_2 = (v_prediction.squeeze() - returns) ** 2

                value_loss = th.mean(th.max(value_loss_1, value_loss_2))

                loss = policy_loss.mean() + self.value_loss_weight * value_loss

                if self.plot:
                    self.pi_loss_history.append(policy_loss.mean().item())
                    self.v_loss_history.append(value_loss.item())
                    self.total_loss_history.append(loss.item())

                    self.expl_var_history.append(explained_variance.item())

                    self.entropy_history.append(entropy.mean().item())

                loss.backward()

                # th.nn.utils.clip_grad_norm_(
                #     self.trainable_parameters, MAX_GRAD_NORM)
                self.optim.step()

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
            # update_network(value_loss, self.optim_v)

    def run_train_loop(self):
        """
        Runs the basic PPO training loop
        """
        if self.plot:
            # Create a plot to show the progress of the training
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
            self.main_ax[1, 1].set_title("Explained Variance")

            self.expl_var_plot, = self.main_ax[1, 1].plot([], [], color="grey")

            # Setup reward plot
            self.main_ax[1, 2].set_autoscale_on(True)
            self.main_ax[1, 2].autoscale_view(True, True, True)
            self.main_ax[1, 2].set_title("Reward per Episode")

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

        for i in range(self.ppo_iterations):
            for eps in range(self.episodes):
                print(
                    f"🎬 Starting {self.env_name} episode {eps + 1}/{self.episodes}")
                self.run_episode(hard_reset=eps == 0)

            # Trim the size of memory buffer:
            if len(self.memories) > self.mem_buffer_size:
                # self.memories.sort(key=lambda mem: mem.total_reward)
                self.memories = self.memories[-self.mem_buffer_size:]
                print(
                    f"⚠️ Trimmed memory buffer to length {self.mem_buffer_size} (worst - {self.memories[0].total_reward} | best - {self.memories[-1].total_reward})")

            self.learn()
            # self.memories.clear()


if __name__ == "__main__":
    rc = RewardsCalculator(
        damage_dealt=1,
    )
    rc.set_time_punishment(-0.5)
    ppo = ProximalPolicyOptimizer(
        # "MineRLPunchCowEz-v0", # Caleb: commented out and replaced with "MineRLPunchCow-v0"
        "MineRLPunchCow-v0",
        "models/foundation-model-1x.model",
        "weights/foundation-model-1x.weights",

        rc=rc,
        ppo_iterations=100,
        episodes=5,
        epochs=4,
        minibatch_size=20,
        lr=0.000025,
        weight_decay=0,
        betas=(0.9, 0.999),
        beta_s=0.999,
        eps_clip=0.1,
        value_clip=0.1,
        value_loss_weight=1.,
        gamma=0.99,
        lam=0.95,
        mem_buffer_size=10000,
        plot=True,
    )

    ppo.run_train_loop()
