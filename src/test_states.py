
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
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
# device = th.device("mps")  # apple silicon


class Testor:
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
            value_loss_weight: float = 0.5,
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
        self.value_loss_weight = value_loss_weight
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

        # This can be adjusted later to only train certain heads
        # Unifying all parameters under one optimizer gives us much more flexibility
        trainable_parameters = self.agent.policy.parameters()

        self.optim = th.optim.Adam(trainable_parameters, lr=lr, betas=betas)

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
        state = self.agent.policy.initial_state(1)

        # I think these are just internal masks that should just be set to false
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
        total_reward = 0

        #TODO Caleb: using num_states, we can collect different states from the net
        max_states = 3
        num_states = 0
        # while not done or num_states <= max_states:
        counter = 0
        while not done:
            counter +=1
            print(counter)
            # Preprocess image

            agent_obs = self.agent._env_obs_to_agent(obs)

            # Basically just adds a dimension to the tensor
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)

            # print(agent_obs["img"].shape)
            # print(dummy_first.shape)
            # print(len(state))
            # print("hi")
            # print(len(state[0]))
            # print(len(state[1]))
            # print(len(state[2]))
            # print(len(state[3]))

            # print(state[1].shape)
            # print(state[2].shape)
            # print(state[3].shape)

            # Need to run this with no_grad or else the gradient descent will crash during training lol
            
            with th.no_grad():
                (pi_h, v_h), state = self.agent.policy.net(
                    agent_obs, state, context={"first": dummy_first})

                # print(state.shape)
                # print("p and v")
                print(pi_h.shape)
                print(v_h.shape)

                # print("state")
                # print(state[0][0].shape)
                print(state[0][1][0].shape)
                # print(state[0][1][1].shape)

                # # print(state[1][0].shape)
                # # print(len(state)) # there are 4 parts of state overall
                # # print(len(state[0]))
                # print(state[1][0].shape)
                # print(state[1][1][0].shape)
                # print(state[1][1][1].shape)

                # print(state[2][0].shape)
                # print(state[2][1][0].shape)
                # print(state[2][1][1].shape)

                # print(state[3][0].shape)
                # print(state[3][1][0].shape)
                # print(state[3][1][1].shape)


                #TODO Caleb: Next we will put heads and state in folder based on Net iteration
                folder_path = f"/home/calebtalley/battle-craft-rl/caleb_states/Net_{num_states}" # the file

                # add fiels to each folder
                # th.save(pi_h, f"{folder_path}/pi_h.pt")
                # th.save(v_h, f"{folder_path}/v_h.pt")
                # th.save(state, f"{folder_path}/state.pt")
                num_states += 1# increment num_states



                
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

            #     # print(action)

                # Get log probability of taking this action given pi
                action_log_prob = self.agent.policy.get_logprob_of_action(
                    pi_distribution, action)

            #     # Process this so the env can accept it
                minerl_action = self.agent._agent_action_to_env(action)

            #     # print(minerl_action)

            #     # Take action step in the environment
                obs, reward, done, info = self.env.step(minerl_action)

            #     # Immediately disregard the reward function from the environment
                reward = self.rc.get_rewards(obs, True)
                total_reward += reward

                memory = Memory(agent_obs, state, pi_h, v_h, action, action_log_prob,
                                reward, 0, done, v_prediction)

                episode_memories.append(memory)
            #     # Finally, render the environment to the screen
            #     # Comment this out if you are boring
                self.env.render()

            # # Update all memories so we know the total reward of its episode
            # # Intuition: memories from episodes where 0 reward was achieved are less valuable
            for mem in episode_memories:
                mem.total_reward = total_reward

        # # Reset the reward calculator once we are done with the episode
        self.rc.clear()

        # # Update internal memory buffer
        self.memories.extend(episode_memories)

        # if self.plot:
        # #     # Updat the reward plot
        #     self.reward_history.append(total_reward)
        #     self.reward_plot.set_ydata(self.reward_history)
        #     self.reward_plot.set_xdata(range(len(self.reward_history)))

        #     self.ax[1, 1].relim()
        #     self.ax[1, 1].autoscale_view(True, True, True)

        #     self.fig.canvas.draw()
        #     self.fig.canvas.flush_events()

        end = datetime.now()
        print(
            f"✅ Episode finished (duration - {end - start} | memories - {len(episode_memories)} | total reward - {total_reward})")


if __name__ == "__main__":
    rc = RewardsCalculator(
    damage_dealt=1
    )
    model_weights_num = input("what model and weights do you want?: ")
    print(f"Using models/foundation-model-{model_weights_num}x.model")
    print(f"Using models/foundation-model-{model_weights_num}x.weights")

    testor = Testor(
        "MineRLPunchCowEz-v0",
        # "MineRLPunchCow-v0",
        f"models/foundation-model-{model_weights_num}x.model",
        f"weights/foundation-model-{model_weights_num}x.weights",
        rc=rc,
        ppo_iterations=100,
        episodes=5,
        epochs=8,
        minibatch_size=48,
        lr=0.000181,
        eps_clip=0.1,

        plot=True
    )
    testor.run_episode()