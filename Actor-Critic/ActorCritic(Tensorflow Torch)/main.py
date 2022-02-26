import gym
import numpy as np
from Agent import Agent
import matplotlib.pyplot as plt

# HYPER PARAMETERS
path = 'AC/'
alpha = 0.0001
gamma = 0.99
episodes = 2000
save_model = False
load_model = True
learn_model = False

env = gym.make('CartPole-v1')
agent = Agent(alpha, gamma, env.action_space.n)
scores = []

if load_model:
    agent.load_checkpoint(path)
    print('Load the checkpoint.............')

for i in range(episodes):
    state = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        score += reward
        if learn_model:
            agent.learn(state, action, reward, next_state, done)
        state = next_state

    scores.append(score)
    avj_score = np.mean(scores[-100:])
    print(f'Episode: {i}, Score: {score}, Average Score: {avj_score}')

    if save_model:
        if i % 100 == 0:
            agent.save_checkpoint(path)
            print('Save the checkpoint..........')



