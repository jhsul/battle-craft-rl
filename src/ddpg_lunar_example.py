from ddpg import Agent
import gym
import numpy as np


env = gym.make('LunarLanderContinuous-v2')

agent = Agent(actor_lr=0.000025, critic_lr=0.00025,
              input_dims=[8], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

score_history = []
for i in range(1000):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        act = agent.choose_action(obs)
        next_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, next_state, int(done))
        # This is a TD method, learn at every step
        # (as opposed to Monte Carlo methods than learn after each episode)
        agent.learn()
        score += reward
        obs = next_state
    
    score_history.append(score)
    print(f'episode {i} score is {round(score, 2)} and 100 game avg is {np.mean(score_history[-100:])}')
    if i % 25 == 0:
        agent.save_models()



