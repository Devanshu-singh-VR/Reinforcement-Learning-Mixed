import gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('Blackjack-v1')
    EPS = 0.05
    GAMMA = 1.0

    agentSumSpace = [i for i in range(4, 22)]
    dealerShowCardSpace = [i+1 for i in range(10)]
    agentAceSpace = [False, True]
    actionSpace = [0, 1]
    stateSpace = []

    Q = {} # for state action pair
    C = {} # C is the sum of relative weights
    for total in agentSumSpace:
        for card in dealerShowCardSpace:
            for ace in agentAceSpace:
                for action in actionSpace:
                    Q[((total, card, ace), action)] = 0
                    C[((total, card, ace), action)] = 0
                stateSpace.append((total, card, ace))

    targetPolicy = {}
    for state in stateSpace:
        values = np.array([Q[(state, a)] for a in actionSpace])
        best = np.random.choice(np.where(values == values.max())[0])
        targetPolicy[state] = actionSpace[best]

    numEpisodes = 1000000
    for i in range(numEpisodes):
        memory = []
        if i % 100000 == 0:
            print('episode: ', i)
        behaviourPolicy = {}
        for state in stateSpace:
            rand = np.random.random()
            if rand < 1 - EPS:
                behaviourPolicy[state] = [targetPolicy[state]] # greedy action
            else:
                behaviourPolicy[state] = actionSpace
        observation = env.reset()
        done = False
        while not done:
            action = np.random.choice(behaviourPolicy[observation])
            observation_, reward, done, info = env.step(action)
            memory.append((observation[0], observation[1],
                           observation[2], action, reward))
            observation = observation_
        memory.append((observation[0], observation[1],
                       observation[2], action, reward))

        G = 0
        W = 1
        last = True # for skipping the terminal state
        for playerSum, dealerCard, usableAce, action, reward in reversed(memory):
            sa = ((playerSum, dealerCard, usableAce), action)
            if last:
                last = False
            else:
                C[sa] += W
                Q[sa] += (W / C[sa]) * (G - Q[sa])
                values = np.array([Q[(state, a)] for a in actionSpace])
                best = np.random.choice(np.where(values == values.max())[0])
                targetPolicy[state] = actionSpace[best]
                if action != targetPolicy[state]:
                    break
                if len(behaviourPolicy[state]) == 1:
                    prob = 1 - EPS
                else:
                    prob = EPS / len(behaviourPolicy[state])

                W *= 1 / prob
            G = GAMMA*G + reward



