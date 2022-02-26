import torch
import numpy as np

class ReplayBuffer(object):
    def __init__(self, buffer_mem, num_states, num_actions):
        self.switch = 0
        self.buffer_mem = buffer_mem
        self.count_mem = 0
        self.num_states = num_states
        self.num_actions = num_actions

        self.state = torch.zeros((buffer_mem, num_states))
        self.action = torch.zeros((buffer_mem, num_actions))
        self.reward = torch.zeros(buffer_mem)
        self.next_state = torch.zeros((buffer_mem, num_states))
        self.terminal = torch.zeros(buffer_mem)

    def load_buffer(self, state, action, reward, next_state, done):
        # this will start rewriting the buffer memory after getting full
        index = self.count_mem % self.buffer_mem

        self.state[index] = torch.tensor(state)
        # one hot vector fot the actions values
        actions = torch.zeros(self.num_actions)
        actions[action] = 1.0
        self.action[index] = actions

        self.reward[index] = torch.tensor(reward)
        self.next_state[index] = torch.tensor(next_state)
        self.terminal[index] = 1 - int(done)

        self.count_mem += 1

    def get_batch(self, batch_size):
        if self.switch == 0 and self.reward[-1] == 0:
            idx = np.random.choice(self.count_mem - batch_size)
        else:
            self.switch = 1
            idx = np.random.choice(self.buffer_mem - batch_size)

        state = self.state[idx: idx+batch_size]
        action = self.action[idx: idx+batch_size]
        reward = self.reward[idx: idx+batch_size]
        next_state = self.next_state[idx: idx+batch_size]
        terminal = self.terminal[idx: idx+batch_size]

        return state, action, reward, next_state, terminal

