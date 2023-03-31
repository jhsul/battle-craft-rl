from ddpg import ExampleAgent
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch

## Not enough to be deterministic, so why even bother...
# np.random.seed(19)
# torch.manual_seed(19)

env = gym.make('LunarLanderContinuous-v2')

agent = ExampleAgent(actor_lr=0.000025, critic_lr=0.00025,
              input_dims=[8], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2, load=False)



score_history = []
for i in range(1000):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        act = agent.choose_action(obs)
        next_agent_obs, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, next_agent_obs, int(done))
        # This is a TD method, learn at every step
        # (as opposed to Monte Carlo methods than learn after each episode)
        agent.learn()
        score += reward
        obs = next_agent_obs
    
    score_history.append(score)
    print(f'episode {i} score is {round(score, 2)} and 100 game avg is {np.mean(score_history[-100:])}')
    if i % 25 == 0:
        agent.save_models()

print(score_history)
plt.plot(score_history)
plt.show()




