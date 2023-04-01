import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import gym


# thanks to Machine Learning with Phil on YouTube for the video guide
# on implementing this algorithm
# https://www.youtube.com/watch?v=6Yd5WnYls_Y

# implement the noise class
# use this to create the exploratory behaviour policy beta based on the
# deterministic action network
class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None) -> None:
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    # calculate the noise (I dont know the math behind it, this is a wrote Copy paste of the formulas)
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
        self.sigma * np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

# Buffer to store <state, action, reward, next_state, terminal> tuples from episodes
# will sample from this to train the networks with mini-batches
class ReplayBuffer:
    def __init__(self, max_size) -> None:
        self.mem_size = max_size
        self.mem_cntr = 0

        self.agent_obs_memory = np.zeros((self.mem_size), dtype=object)
        self.next_agent_obs_memory = np.zeros((self.mem_size), dtype=object)
        self.action_memory = np.zeros((self.mem_size), dtype=object)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.vpt_state_memory = np.zeros(self.mem_size, dtype=object)

    def __getitem__(self, key):
        agent_obs = self.agent_obs_memory[key]
        next_agent_obs = self.next_agent_obs_memory[key]
        reward = self.reward_memory[key]
        action = self.action_memory[key]
        terminal = self.terminal_memory[key] 
        vpt_state = self.vpt_state_memory[key]

        return agent_obs, action, reward, next_agent_obs, terminal, vpt_state
    
    def __len__(self):
        return self.mem_cntr

    # state in this case is basically the obs
    def store_transition(self, agent_obs, action, reward, next_agent_obs, done, vpt_state):
        # wrapping index to override oldest memory when you reach limit
        index = self.mem_cntr % self.mem_size

        self.agent_obs_memory[index] = agent_obs
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_agent_obs_memory[index] = next_agent_obs
        self.terminal_memory[index] = 1-done
        self.vpt_state_memory[index] = vpt_state

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):

        # generate indices for random indexing
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        # grab random samples from the buffer
        states = self.agent_obs_memory[batch]
        next_states = self.next_agent_obs_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminals = self.terminal_memory[batch]
        vpt_states = self.vpt_state_memory[batch]

        return states, actions, rewards, next_states, terminals, vpt_states


# Example critic network for DDPG that has 2 fully connected hidden layers
# TODO figure how to integrate this with MineRL...
class ExampleCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ExampleCriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')

        # define NN layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # not sure why the paramters are initialized like this
        # sure it's in the paper somewhere, but not that relevent if we are tuning VPT
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        # add a batch norm layer
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        # add a batch norm layer
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # a linear layer for processing the action value
        self.action_value = nn.Linear(self.n_actions, fc2_dims)

        # q network as in Q learning!
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)


        ## other important NN initialization steps for pytorch
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        # actually pass data through the network layers
        # notice how we manually call activation functions
        # instead of defining them with layers
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        # double relu on action value? weird?
        action_value = F.relu(self.action_value(action))
        # this is how states and actions are integrated into this network
        # we just add them together after passing them through the state
        # and action layers, interesting
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value
    
    def save_checkpoint(self):
        print("...saving checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ExampleActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name,
                chkpt_dir='tmp/ddpg'):
        super(ExampleActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
    
        # define NN layers, similar to ExampleCriticNetwork
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # not sure why the paramters are initialized like this
        # sure it's in the paper somewhere, but not that relevent if we are tuning VPT
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        # add a batch norm layer
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        # add a batch norm layer
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # mu is the (deterministic) action network
        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)


        ## more settings
        ## other important NN initialization steps for pytorch
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):

        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # bound between -1 and 1, multiple by bounds of actions later?
        x = T.tanh(self.mu(x))


        return x
    
    def save_checkpoint(self):
        print("...saving checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ExampleAgent():
    # tau is a hyperparameter for updating the target network
    # gamma is the discount factor
    def __init__(self, actor_lr, critic_lr, input_dims, tau, env, gamma=0.99, 
                 n_actions=2, max_size=1000000,
                 layer1_size=400, layer2_size=300, batch_size=64, load=False) -> None:
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size


        ## change this out with the MineRL heads
        self.actor = ExampleActorNetwork(actor_lr, input_dims, layer1_size, layer2_size,
                                          n_actions=n_actions, name="actor")
        
        self.target_actor = ExampleActorNetwork(actor_lr, input_dims, layer1_size, layer2_size,
                                          n_actions=n_actions, name="target_actor")
        
        self.critic = ExampleCriticNetwork(critic_lr, input_dims, layer1_size, layer2_size,
                                           n_actions=n_actions, name="critic")
        
        self.target_critic = ExampleCriticNetwork(critic_lr, input_dims, layer1_size, layer2_size,
                                           n_actions=n_actions, name="target_critic")
        
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

        if load:
            self.load_models()

    def choose_action(self, obs):
        # set eval mode for batch norm
        self.actor.eval()
        obs = T.tensor(obs, dtype=T.float).to(self.actor.device)
        mu = self.actor(obs).to(self.actor.device)
        # action with noise
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()
    
    def remember(self, state, actoin, reward, next_state, done):
        self.memory.store_transition(state, actoin, reward, next_state, done)

    def learn(self):
        # start learning only when you have enough samples
        # inside of the replay buffer! (non-obvious imp. detail)
        if self.memory.mem_cntr < self.batch_size:
            return
        
        # load in all of the data from the replay buffer
        state, action, reward, next_state, done = \
            self.memory.sample_buffer(self.batch_size)
        
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        # set networks to eval mode
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # calculate relevent network values
        target_actions = self.target_actor.forward(next_state)
        next_critic_value = self.target_critic.forward(next_state, target_actions)
        critic_value = self.critic.forward(state, action)

        # calculate bellman equation using network values
        # TODO vectorize this better
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*next_critic_value[j]*done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        # now train the critic
        # still not entirely sure how training in Pytorch works
        # or what each method call exactly does
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # train the actor
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()

        # the gradient of the actor is just the forward pass of the critic?
        # some result from the paper... (I think that is what this says?)
        actor_loss = -self.critic.forward(state, mu)
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




# This wrapper class is to normalize the values of the action space
# Pulled from this https://github.com/ghliu/pytorch-ddpg/blob/master/normalized_env.py implementation
# of DDPG on github
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)