
import pickle
import sys
from typing import List
import gym
import torch as th
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from tqdm import tqdm
from rewards import RewardsCalculator
from memory import Memory, MemoryDataset
from util import to_torch_tensor, normalize, safe_reset, hard_reset

sys.path.insert(0, "vpt")  # nopep8

from agent import MineRLAgent  # nopep8
from lib.tree_util import tree_map  # nopep8

# For debugging purposes
# th.autograd.set_detect_anomaly(True)
# device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
device = th.device("mps")  # apple silicon


class ProximalPolicyOptimizer:
    def __init__(
            self,
            env_name: str,
            model_path: str,
            weights_path: str,

            # A custom reward function to be used
            rc: RewardsCalculator = None,

            # Hyperparameters
            ppo_iterations=5,
            episodes: int = 3,
            epochs: int = 8,
            minibatch_size: int = 10,
            lr: float = 1e-4,
            betas: tuple = (0.9, 0.999),
            beta_s: float = 0.01,
            eps_clip: float = 0.1,
            value_clip: float = 0.4,
            gamma: float = 0.99,
            lam: float = 0.95,
            # tau: float = 0.95,

            # Optional: plot stuff
            plot: bool = False,

            mem_buffer_size: int = 50000,

    ):
        self.env_name = env_name
        self.env = gym.make(env_name)

        # Set the rewards calcualtor
        if rc is None:
            # Basic default reward function
            rc = RewardsCalculator(
                damage_dealt=1,
                damage_taken=-1,
            )
        self.rc = rc

        # Load hyperparameters unchanged
        self.ppo_iterations = ppo_iterations
        self.episodes = episodes
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.betas = betas
        self.beta_s = beta_s
        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.gamma = gamma
        self.lam = lam
        # self.tau = tau

        self.plot = plot

        self.mem_buffer_size = mem_buffer_size

        if self.plot:
            self.pi_loss_history = []
            self.v_loss_history = []

            self.entropy_history = []

            self.surr1_history = []
            self.surr2_history = []

            self.reward_history = []

        # Load the agent parameters from the weight files
        agent_parameters = pickle.load(open(model_path, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

        self.agent = MineRLAgent(self.env, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(weights_path)

        self.optim_pi = th.optim.Adam(
            self.agent.policy.pi_head.parameters(), lr=lr, betas=betas)

        self.optim_v = th.optim.Adam(
            self.agent.policy.value_head.parameters(), lr=lr, betas=betas)

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
        # In learn(), we will use a variable batch size
        state = self.agent.policy.initial_state(1)

        # IDK what this is!
        dummy_first = th.from_numpy(np.array((False,))).to(device)

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
        total_reward = 0

        while not done:
            # Preprocess image
            agent_obs = self.agent._env_obs_to_agent(obs)

            # Run the full model to get both heads and the
            # new hidden state
            # pi_distribution, v_prediction, state = self.agent.policy.get_output_for_observation(
            #     agent_obs,
            #     state,
            #     dummy_first
            # )
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)
            first = dummy_first.unsqueeze(1)

            # Need to run this with no_grad or else the gradient descent will crash during training lol
            with th.no_grad():
                (pi_h, v_h), state = self.agent.policy.net(
                    agent_obs, state, context={"first": first})

                pi_distribution = self.agent.policy.pi_head(pi_h)
                v_prediction = self.agent.policy.value_head(v_h)

                # print(pi_distribution)
                # print(pi_distribution["camera"].shape)
                # print(pi_distribution["buttons"].shape)
                # print(v_prediction)

            # print(pi_distribution)
            # print(policy.get_logprob_of_action(pi_distribution, None))

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

            # print(minerl_action)

            # Take action step in the environment
            obs, reward, done, info = self.env.step(minerl_action)

            # Immediately disregard the reward function from the environment
            reward = self.rc.get_rewards(obs, True)
            total_reward += reward

            memory = Memory(pi_h, v_h, action, action_log_prob,
                            reward, 0, done, v_prediction)

            episode_memories.append(memory)
            # Finally, render the environment to the screen
            # Comment this out if you are boring
            self.env.render()

        # Update all memories so we know the total reward of its episode
        # Intuition: memories from episodes where 0 reward was achieved are less valuable
        for mem in episode_memories:
            mem.total_reward = total_reward

        # Reset the reward calculator once we are done with the episode
        self.rc.clear()

        # Update internal memory buffer
        self.memories.extend(episode_memories)

        if self.plot:
            # Updat the reward plot
            self.reward_history.append(total_reward)
            self.reward_plot.set_ydata(self.reward_history)
            self.reward_plot.set_xdata(range(len(self.reward_history)))

            self.ax[1, 1].relim()
            self.ax[1, 1].autoscale_view(True, True, True)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        end = datetime.now()
        print(
            f"✅ Episode finished (duration - {end - start} | memories - {len(episode_memories)} | total reward - {total_reward})")

    def learn(self):
        # TODO: calcualte generalized advantage estimate
        # IDK if that is just for PPG or what, but it looked scary

        data = MemoryDataset(self.memories)
        dl = DataLoader(data, batch_size=self.minibatch_size, shuffle=True)

        # if self.plot:
        #     pi_loss_history = []
        #     v_loss_history = []

        for _ in tqdm(range(self.epochs), desc="epochs"):

            # Shuffle the memories
            # Note: These are batches! Not individual samples
            for pi_h, v_h, actions, old_action_log_probs, rewards, total_rewards, dones, v_old in dl:
                # Run the model on the batch using the latent state output
                pi_distribution = self.agent.policy.pi_head(pi_h)
                v_prediction = self.agent.policy.value_head(v_h)

                masks = list(map(lambda d: 1-float(d), dones))

                returns = []
                gae = 0

                for i in reversed(range(len(rewards))):
                    # hacky but necessary since we don't have "next_state"
                    v_next = v_old[i + 1] if i != len(rewards) - 1 else 0

                    delta = rewards[i] + self.gamma * \
                        v_next * masks[i] - v_old[i]
                    gae = delta + self.gamma * self.lam * masks[i] * gae
                    returns.insert(0, gae + v_old[i])

                # print(rewards)
                # Overwrite the rewards now
                rewards = th.tensor(returns).float().to(device)

                # print(rewards)
                # Get log probs
                action_log_probs = self.agent.policy.get_logprob_of_action(
                    pi_distribution, actions)

                entropy = self.agent.policy.pi_head.entropy(pi_distribution)

                # Calculate clipped surrogate objective
                ratios = (action_log_probs - old_action_log_probs).exp()
                advantages = normalize(rewards - v_prediction.detach())
                surr1 = ratios * advantages
                surr2 = ratios.clamp(
                    1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - th.min(surr1, surr2) - self.beta_s * entropy

                # print(type(policy_loss))
                # print(policy_loss.shape)

                # Calculate clipped value loss
                value_clipped = v_old + \
                    (v_prediction - v_old).clamp(-self.value_clip, self.value_clip)

                value_loss_1 = (value_clipped.squeeze() - rewards) ** 2
                value_loss_2 = (v_prediction.squeeze() - rewards) ** 2

                value_loss = th.mean(th.max(value_loss_1, value_loss_2))
                # Calculate clipped value loss
                # value_loss = clipped_value_loss(
                #     v_prediction, rewards, old_values, self.value_clip)

                # if self.plot:
                #     self.pi_loss_history.append(policy_loss.mean().item())
                #     self.v_loss_history.append(value_loss.item())

                #     self.entropy_history.append(entropy.mean().item())

                # Update the policy network
                self.optim_pi.zero_grad()
                policy_loss.mean().backward()
                self.optim_pi.step()

                # Update the value network
                self.optim_v.zero_grad()
                value_loss.backward()
                self.optim_v.step()

            # Update plot at the end of every epoch
            if self.plot:
                self.pi_loss_history.append(policy_loss.mean().item())
                self.v_loss_history.append(value_loss.item())

                self.entropy_history.append(entropy.mean().item())

                # Update the loss plots
                self.pi_loss_plot.set_ydata(self.pi_loss_history)
                self.pi_loss_plot.set_xdata(
                    list(range(len(self.pi_loss_history))))

                # Update policy loss plot
                self.ax[0, 0].relim()
                self.ax[0, 0].autoscale_view(True, True, True)

                # Update value loss plot

                self.v_loss_plot.set_ydata(self.v_loss_history)
                self.v_loss_plot.set_xdata(
                    list(range(len(self.v_loss_history))))
                self.ax[0, 1].relim()
                self.ax[0, 1].autoscale_view(True, True, True)

                # Update the entropy plot
                self.entropy_plot.set_ydata(self.entropy_history)
                self.entropy_plot.set_xdata(range(len(self.entropy_history)))

                self.ax[1, 0].relim()
                self.ax[1, 0].autoscale_view(True, True, True)

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            # update_network(value_loss, self.optim_v)

    def run_train_loop(self):
        """
        Runs the basic PPO training loop
        """
        if self.plot:
            # Create a plot to show the progress of the training
            plt.ion()
            self.fig, self.ax = plt.subplots(2, 2, figsize=(10, 8))

            # Set up policy loss plot
            self.ax[0, 0].set_autoscale_on(True)
            self.ax[0, 0].autoscale_view(True, True, True)

            self.ax[0, 0].set_title("Policy Loss")

            self.pi_loss_plot, = self.ax[0, 0].plot(
                [], [], color="blue")

            # Setup value loss plot
            self.ax[0, 1].set_autoscale_on(True)
            self.ax[0, 1].autoscale_view(True, True, True)

            self.ax[0, 1].set_title("Value Loss")

            self.v_loss_plot, = self.ax[0, 1].plot(
                [], [], color="orange")

            # self.ax[0, 0].legend(loc="upper right")

            # Setup entropy plot
            self.ax[1, 0].set_autoscale_on(True)
            self.ax[1, 0].autoscale_view(True, True, True)
            self.ax[1, 0].set_title("Entropy")

            self.entropy_plot, = self.ax[1, 0].plot([], [], color="green")

            # Setup reward plot
            self.ax[1, 1].set_autoscale_on(True)
            self.ax[1, 1].autoscale_view(True, True, True)
            self.ax[1, 1].set_title("Reward per Episode")

            self.reward_plot,  = self.ax[1, 1].plot([], [], color="red")

        for i in range(self.ppo_iterations):
            for eps in range(self.episodes):
                print(
                    f"🎬 Starting {self.env_name} episode {eps + 1}/{self.episodes}")
                self.run_episode(hard_reset=eps == 0)

            # Trim the size of memory buffer:
            if len(self.memories) > self.mem_buffer_size:
                self.memories.sort(key=lambda mem: mem.total_reward)
                self.memories = self.memories[-self.mem_buffer_size:]
                print(
                    f"⚠️ Trimmed memory buffer to length {self.mem_buffer_size} (worst - {self.memories[0].total_reward} | best - {self.memories[-1].total_reward})")

            self.learn()
            # self.memories.clear()


if __name__ == "__main__":
    rc = RewardsCalculator(
        damage_dealt=1
    )
    ppo = ProximalPolicyOptimizer(
        "MineRLPunchCow-v0",
        "models/foundation-model-2x.model",
        "weights/foundation-model-2x.weights",

        rc=rc,
        ppo_iterations=100,
        episodes=10,
        epochs=12,
        minibatch_size=48,
        lr=0.00025,
        eps_clip=0.1,

        plot=True
    )

    ppo.run_train_loop()
