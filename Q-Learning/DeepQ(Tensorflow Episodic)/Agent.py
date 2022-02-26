import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from ReplayMemory import Buffer
import numpy as np

# using epsilon greedy for exploit explore delema
class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon,
                 batch_size, input_dims, epsilon_dec=0.996,
                 epsilon_end=0.01, mem_size=1000000, save_dir='dqn.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.save_dir = save_dir

        self.memory = Buffer(mem_size, input_dims, n_actions, discrete=True)

        self.q_eval = self.build_dqn(alpha, n_actions, input_dims, 256, 256)

    def build_dqn(self, lr, n_actions, input_dims, fc1_dims, fc2_dims):
        model = Sequential([
            Dense(fc1_dims, input_shape=(input_dims,)),
            Activation('relu'),
            Dense(fc2_dims),
            Activation('relu'),
            Dense(n_actions)
        ])

        model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # epsilon actions
    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.q_eval.predict(state)
            action = np.argmax(action)

        return action

    def learn(self):
        # here we wanna start learning after filling the level of buffer size
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = (
            self.memory.sample_buffer(self.batch_size)
        )

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = (
                reward + self.gamma*np.max(q_next, axis=1)*done) # for the q_value which has more target

        self.q_eval.fit(state, q_target, verbose=0)

        # decreasing the epsilon value over the time
        self.epsilon = (
            self.epsilon*self.epsilon_dec
            if self.epsilon > self.epsilon_min
            else self.epsilon_min
        )

    def save_model(self):
        self.q_eval.save(self.save_dir)

    def load_model(self):
        self.q_eval = load_model(self.save_dir)
