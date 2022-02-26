from Agent import Agent
import numpy as np
import torch
import gym

# HYPERPARAMETERS
path = 'dqn.pth.tar'
load_model = True
save_model = False
episode = 500
alpha = 0.0005
gamma = 0.99
batch_size = 64
num_actions = 4
num_states = 8
epsilon = 0

# Gym Environment and Agent
env = gym.make('LunarLander-v2')
scores = []
agent = Agent(alpha, gamma, batch_size, num_actions,
              num_states, epsilon)

# Loading the Model
if load_model:
    agent.load_checkpoint(torch.load(path))

for i in range(episode):
    done = False
    score = 0
    state = env.reset()

    # Saving the Model
    if i % 100 == 0:
        checkpoint = {'model_dict': agent.model.state_dict(), 'optimizer_dict': agent.optim.state_dict()}
        if save_model:
            agent.save_checkpoint(checkpoint, path, i)
        
    while not done:
        env.render()
        action = agent.choose_action(state)
        if type(action) == torch.Tensor:
            action = action.item()
        next_state, reward, done, info = env.step(action)
        score += reward
        agent.loading_observation(state, action, reward, next_state, done)
        state = next_state
        #agent.learn()

    scores.append(score)
    avj_score = np.mean(scores[max(0, i-100):(i+1)])

    print(f'Episode: {i}, Score: {score}, Average Score: {avj_score}')
