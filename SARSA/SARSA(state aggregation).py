import numpy as np
import matplotlib.pyplot as plt
import gym

# get the discrete spaces
cart_position = np.linspace(-2.4, 2.4, 10)
cart_velocity = np.linspace(-4, 4, 10)
pole_angle = np.linspace(-0.2095, 0.2095, 10)
pole_angular_velocity = np.linspace(-4, 4, 10)

def get_state(state):
    c_pos, c_val, p_le, p_av = state
    # digitize will provide the index of the discrete
    # space where the value lies close to it. Like the state aggregation
    c_pos = np.digitize(c_pos, cart_position)
    c_val = np.digitize(c_val, cart_velocity)
    p_le = np.digitize(p_le, pole_angle)
    p_av = np.digitize(p_av, pole_angular_velocity)
    return c_pos, c_val, p_le, p_av

# store the state with state aggregation
states = []
for i in range(len(cart_position)+1):
    for j in range(len(cart_velocity)+1):
        for k in range(len(pole_angle)+1):
            for l in range(len(pole_angular_velocity)+1):
                states.append((i, j, k, l))

# Choose action
def max_action(Q, state):
    value = np.array([Q[state, action] for action in range(2)])
    action = np.argmax(value)
    return action

# Store the state action pair
Q = {}
for s in states:
    for a in range(2):
        Q[s, a] = 0

# HYPER PARAMETERS
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_descent = 0.9999
min_epsilon = 0.01
episodes = 20000
scores = []
avj_score = []
ep_list = []
env = gym.make('CartPole-v0')

# Start the episode
for i in range(episodes):
    state = env.reset()
    s = get_state(state)
    rand = np.random.random()
    a = (env.action_space.sample()
         if rand < epsilon
         else max_action(Q, s)
         )
    done = False
    score = 0
    # Algo
    while not done:
        next_state, reward, done, info = env.step(a)
        s_ = get_state(next_state)
        rand = np.random.random()
        a_ = (env.action_space.sample()
              if rand < epsilon
              else max_action(Q, s_)
              )
        score += reward
        Q[s, a] = Q[s, a] + alpha*(reward + gamma*Q[s_, a_] - Q[s, a])
        s = s_
        a = a_

    scores.append(score)
    avj_score.append(np.mean(scores[-100:]))
    ep_list.append(epsilon*200)
    epsilon = (epsilon*epsilon_descent
               if epsilon > min_epsilon
               else min_epsilon
               )
    print(f'Episode: {i}, Score:{score}, Average Score: {np.mean(scores[-100:])}')

steps = [i for i in range(episodes)]
plt.title('SARSA')
plt.xlabel('steps')
plt.ylabel('score')
plt.scatter(steps, avj_score, color='blue')
plt.plot(steps, ep_list, color='red')
plt.show()