from Agent import Agent
import gym

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 20
    agent = Agent(alpha=0.0005, gamma=0.99, epsilon=0.0,
                  input_dims=8, n_actions=4, mem_size=1000000,
                  batch_size=64)

    agent.load_model()

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_

        print(f'episode: {i}, score: {score}')

