import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Agent import Agent
import numpy as np
import gym

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 500
    agent = Agent(alpha=0.0005, gamma=0.99, epsilon=1.0,
                  input_dims=8, n_actions=4, mem_size=1000000,
                  batch_size=64)

    #agent.load_model()

    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            #agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avj_score = np.mean(scores[max(0, i-100):(i+1)])
        print(f'episode: {i}, score: {score}, avj_score: {avj_score}')

        #if i % 10 == 0:
        #    agent.save_model()

