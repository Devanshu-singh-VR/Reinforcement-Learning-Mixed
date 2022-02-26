import matplotlib.pyplot as plt
from Agent import ACAgent
import numpy as np
import gym
import torch

# HYPER PARAMETERS
path = 'ac.pth.tar'
alpha = 0.0005
gamma = 0.996
load_model = True
save_model = False
learn_model = False
num_states = 4
num_actions = 2
episodes = 2000
scores = []
ajv_scores = []
epi = []

# Environment and Agent
env = gym.make('CartPole-v0')
agent = ACAgent(alpha, gamma, num_actions, num_states)

# Loading the Model
if load_model:
    agent.load_checkpoint(torch.load(path))

# Algo
for i in range(episodes):
    state = env.reset()
    score = 0
    done = False

    # Saving the Model
    if i % 100 == 0:
        checkpoint = {'model_dict': agent.network.state_dict(), 'optimizer_dict': agent.optimizer.state_dict()}
        if save_model:
            agent.save_checkpoint(checkpoint, path, i)

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
    epi.append(i)
    ajv_scores.append(avj_score)
    print(f'Episode: {i}, Score: {score}, Average Score; {avj_score}')

plt.scatter(epi, ajv_scores)
plt.show()