import gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('Blackjack-v1')
    EPS = 0.05
    GAMMA = 1.0

    Q = {}
    agentSumSpace = [i for i in range(1, 22)]
    dealerShowCardSpace = [i+1 for i in range(10)]
    agentAceSpace = [False, True]
    actionSpace = [0, 1]

    stateSpace = []
    returns = {}
    pairsVisited = {}
    for total in agentSumSpace:
        for card in dealerShowCardSpace:
            for ace in agentAceSpace:
                for action in actionSpace:
                    Q[((total, card, ace), action)] = 0 # state action pair
                    returns[((total, card, ace), action)] = 0
                    pairsVisited[((total, card, ace), action)] = 0
                stateSpace.append((total, card, ace))

    policy = {}
    for state in stateSpace:
        policy[state] = np.random.choice(actionSpace)

    numEpisodes = 1000000
    for i in range(numEpisodes):
        stateActionsReturns = []
        memory = []
        if i % 100000 == 0:
            print('episode: ', i)
        observation = env.reset()
        done = False
        while not done:
            action = policy[observation]
            observation_, reward, done, info = env.step(action)
            memory.append((observation[0], observation[1],
                           observation[2], action, reward))
            observation = observation_
        memory.append((observation[0], observation[1],
                       observation[2], action, reward))

        # calculating the estimated reward from the reversed memory
        G = 0
        last = True
        for playerSum, dealerCard, usableAce, action, reward in reversed(memory):
            if last:
                last = False
            else:
                stateActionsReturns.append((playerSum, dealerCard, usableAce, action, G))
            G = GAMMA*G + reward

        stateActionsReturns.reverse()

        for playerSum, dealerCard, usableAce, action, G in stateActionsReturns:
            sa = ((playerSum, dealerCard, usableAce), action) # state action tuple
            if True:
                pairsVisited[sa] += 1
                # new_estimate = old_estimate + 1/n * ( target - old_estimate )
                returns[(sa)] += (1 / pairsVisited[(sa)]) * (G - returns[(sa)])
                Q[sa] = returns[sa]
                rand = np.random.random()
                if rand < 1 - EPS: # for greedy action
                    state = (playerSum, dealerCard, usableAce)
                    values = np.array([Q[(state, a)] for a in actionSpace])
                    best = np.random.choice(np.where(values == values.max())[0])
                    policy[state] = actionSpace[best]
                else: # for the random action
                    policy[state] = np.random.choice(actionSpace)

        if EPS - 1e-7 > 0:
            EPS -= 1e-7
        else:
            EPS = 0

    numEpisodes = 1000
    rewards = np.zeros(numEpisodes)
    totalReward = 0
    wins = 0
    losses = 0
    draws = 0

    for i in range(numEpisodes):
        observation = env.reset()
        done = False
        while not done:
            action = policy[observation]
            observation_, reward, done, info = env.step(action)
            observation = observation_
        totalReward += reward
        rewards[i] = totalReward

        if reward >= 1:
            wins += 1
        elif reward == 0:
            draws += 1
        elif reward == -1:
            losses += 1

    wins /= numEpisodes
    losses /= numEpisodes
    draws /= numEpisodes

    print('win rate: ', wins, 'loss rate: ', losses, 'draw rate: ', draws)
    plt.plot(rewards)
    plt.show()

