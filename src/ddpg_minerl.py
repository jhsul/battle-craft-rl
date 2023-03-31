from ddpg import NormalizedEnv, ReplayBuffer, OUActionNoise, ExampleAgent
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch as T
import pickle
import sys
import torch.nn.functional as F
from datetime import datetime
from util import safe_reset
from rewards import RewardsCalculator
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, "vpt")  # nopep8
from lib.tree_util import tree_map  # nopep8
from agent import MineRLAgent  # nopep8

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class ReplayBufferDataset(Dataset):
    """
    This is a dataset of memory objects (potentially multiple episodes!)
    This is to be used with the PyTorch DataLoader
    """

    def __init__(self, memories):
        self.memories: ReplayBuffer = memories

    def __len__(self):
        return len(self.memories)

    def __getitem__(self, idx):
        return self.memories[idx]



class DDPG_MineRLAgent():
    # tau is a hyperparameter for updating the target network
    # gamma is the discount factor
    def __init__(self, actor_lr, critic_lr, input_dims, tau, env ,gamma=0.99, max_size=10000,
                 batch_size=64, load=False, betas: tuple = (0.9, 0.999),
                 model_path="./models/foundation-model-2x.model", 
                 weights_path="./weights/foundation-model-2x.weights"):
        
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims)
        self.batch_size = batch_size

       # Load the agent parameters from the weight files
        agent_parameters = pickle.load(open(model_path, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

        self.agent = MineRLAgent(self.env, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(weights_path)

        self.target_agent = MineRLAgent(self.env, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.target_agent.load_weights(weights_path)

        self.actor_optim = T.optim.Adam(
            self.agent.policy.pi_head.parameters(), lr=actor_lr, betas=betas)

        self.critic_optim = T.optim.Adam(
            self.agent.policy.value_head.parameters(), lr=critic_lr, betas=betas)

        ## The actor and critic for this are the head of the VPT model
        self.actor = self.agent.policy.pi_head
        self.target_actor = self.target_agent.policy.pi_head
        
        self.critic = self.agent.policy.value_head
        self.target_critic = self.target_agent.policy.value_head
        
        # noise only for the camera (continuous action)
        # TODO how to add noise to the buttons?
        self.noise = OUActionNoise(mu=np.zeros(2))

        self.update_network_parameters(tau=1)

        if load:
            self.load_models()

    def choose_action(self, obs, first, state):

        # This is a very cheap way of using DDPG, just using argmax to make this deterministic
        # Run the forward pass through the model, first the base then the heads
        with T.no_grad():
            # pass the obs through the base model
            # we probably dont need to do this separately
            (pi_h, v_h), state = self.agent.policy.net(
                agent_obs, state, context={"first": first})

            # pass what we got from the base model into the policy head
            pi_distribution = self.agent.policy.pi_head(pi_h)

        # get the action from the pi head deterministically for DDPG
        action = self.actor.sample(
                pi_distribution, deterministic=True)

        # process the action for input into the env
        minerl_action = self.agent._agent_action_to_env(action)

        # add noise to the action
        minerl_action['camera'] = minerl_action['camera'] + self.noise()

        self.actor.train()
        return minerl_action, state
    
    def remember(self, state, action, reward, next_state, done, vpt_state):
        # make the torchc tensors numpy arrays for storage
        self.memory.store_transition(state['img'].detach().numpy(), 
                                     action, reward, 
                                     next_state['img'].detach().numpy(), 
                                     done, vpt_state)

    def learn(self):
        # start learning only when you have enough samples
        # inside of the replay buffer! (non-obvious imp. detail)
        if self.memory.mem_cntr < self.batch_size:
            return
        

        data = ReplayBufferDataset(self.memory)
        dl = DataLoader(data, self.batch_size)

        # set networks to eval mode
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        for agent_obs, action, reward, next_agent_obs, done, vpt_state in dl:

            dummy_first = T.from_numpy(np.array((False,))).to(device)
            first = dummy_first.unsqueeze(1)
            # get the target critic value
            (pi_h, v_h), state = self.agent.policy.net(
                next_agent_obs, vpt_state, context={"first": first})

            next_critic_value = self.target_critic.forward(v_h)
            
            # get the regular critic value
            # TODO this critic value should probably just be in memory
            # (or not because we are updating as we go?)
            (pi_h, v_h), state = self.agent.policy.net(
                agent_obs, state, context={"first": first})
            critic_value = self.critic.forward(v_h)

            # calculate bellman equation using network values
            target = reward + self.gamma*next_critic_value*done

            # now train the critic
            self.critic.train()
            self.critic_optim.zero_grad()
            critic_loss = F.mse_loss(target, critic_value)
            critic_loss.backward()
            self.critic_optim.step()

            # train the actor
            self.critic.eval()
            self.actor.optimizer.zero_grad()
            mu = self.actor.forward(agent_obs)
            self.actor.train()

            # the gradient of the actor is just the forward pass of the critic?
            # some result from the paper... (I think that is what this says?)
            actor_loss = -self.critic.forward(agent_obs, mu)
            actor_loss = T.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            # update the target network gradually
            self.update_network_parameters()

    # this is for updating the target networks to lag behind the ones we are training
    # tau is small, near 0
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # get all of the parameters from the networks
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        # make these a dict for iteration
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        # these for loops are "averaging" the currect critic/actor values
        # into their respective target networks
        # TODO vectorize this better? at all?
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone()+\
                        (1-tau)*target_critic_state_dict[name].clone()
            
        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone()+\
                        (1-tau)*target_actor_state_dict[name].clone()
            
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()       







def train():
    env_name = 'MineRLPunchCow-v0'
    # May need to normalize action values? may not
    env = gym.make(env_name)

    # not sure how well DDPG works with a mostly discrete action space, guess we'll see!
    ddpg = DDPG_MineRLAgent(actor_lr=0.000025, critic_lr=0.00025,
                input_dims=[128,128,3], tau=0.001, env=env,
                batch_size=64, load=False)
    
    rc = RewardsCalculator(
        damage_dealt=1
    )

    score_history = []
    episodes = 100
    for i in range(episodes):
        print(f"🎬 Starting {env_name} episode {i + 1}/{episodes}")
        start = datetime.now()
        done = False
        score = 0

        # IDK what this is!
        dummy_first = T.from_numpy(np.array((False,))).to(device)
        raw_obs, env = safe_reset(env)


        agent_obs = ddpg.agent._env_obs_to_agent(raw_obs)
        agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)

        # Initialize the hidden state vector
        # Note that batch size is 1 here because we are only running the agent
        # on a single episode
        # In learn(), we will use a variable batch size
        state = ddpg.agent.policy.initial_state(1)

        while not done:
            # Process the observation because MineRL reasons...
            agent_obs = ddpg.agent._env_obs_to_agent(raw_obs)
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)
            first = dummy_first.unsqueeze(1)

            # Choose the action deterministically
            act, new_state = ddpg.choose_action(agent_obs, first, state)
            next_agent_obs, reward, done, info = env.step(act)
        
            # Use our custom reward calculator instead
            reward = rc.get_rewards(next_agent_obs)
            raw_obs = next_agent_obs
            next_agent_obs = ddpg.agent._env_obs_to_agent(next_agent_obs)

            ddpg.remember(agent_obs, act, reward, 
                          next_agent_obs, int(done), state)

            # This is a TD method, learn at every step
            # (as opposed to Monte Carlo methods than learn after each episode)
            ddpg.learn()
            score += reward
            state = new_state
            
            # Finally, render the environment to the screen
            # Comment this out if you are boring
            env.render()
        
        # Reset the RewardsCalculator after the episode
        rc.clear()
        score_history.append(score)
        end = datetime.now()
        print(
            f"✅ Episode finished (duration - {end - start} | total reward - {score})")

    print(score_history)
    plt.plot(score_history)
    plt.show()


if __name__ == "__main__":
    train()