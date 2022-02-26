import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from Buffer import ReplayBuffer
from Network import DQNetwork

class Agent(object):
    def __init__(self, alpha, gamma, batch_size,
                 num_actions, num_states, epsilon,
                 epsilon_descent=0.996, epsilon_min=0.01,
                 buffer_size=1000000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_states = num_states
        self.epsilon = epsilon
        self.epsilon_descent = epsilon_descent
        self.epsilon_min = epsilon_min

        # Buffer FX and Network
        self.buffer = ReplayBuffer(buffer_size, num_states, num_actions)
        self.model = DQNetwork(num_states, num_actions)

        # optimizer and loss function
        self.optim = optim.Adam(self.model.parameters(), lr=alpha)
        self.loss = nn.MSELoss()

    def choose_action(self, state):
        state = torch.tensor(state)
        random = np.random.random()
        if random < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = self.model(state.unsqueeze(0))
            action = torch.argmax(action, dim=1)

        return action

    def loading_observation(self, state, action, reward, next_state, done):
        self.buffer.load_buffer(state, action, reward, next_state, done)

    def learn(self):
        if self.buffer.count_mem <= self.batch_size:
            return

        # Collecting the batch
        state, action, reward, next_state, done = (
            self.buffer.get_batch(self.batch_size)
        )

        # q_value
        q_value = self.model(state)
        next_q_value = self.model(next_state)

        # target value for the particular action which has taken in the state
        target = q_value.clone()
        batches = torch.arange(target.shape[0])
        max_q, idx = torch.max(next_q_value, dim=1)
        target[batches, torch.argmax(action, dim=1)] = (
            reward + self.gamma*max_q*done
        )

        # Training
        self.optim.zero_grad()
        loss = self.loss(q_value, target)
        loss.backward()
        self.optim.step()

        # Epsilon descent
        self.epsilon = (self.epsilon*self.epsilon_descent
                        if self.epsilon > self.epsilon_min
                        else self.epsilon_min
                        )

    def save_checkpoint(self, state, path, epoch):
        print('save the checkpoint ', epoch)
        torch.save(state, path)

    def load_checkpoint(self, saved_checkpoint):
        print('loading the checkpoint')
        self.model.load_state_dict((saved_checkpoint['model_dict']))
        self.optim.load_state_dict(saved_checkpoint['optimizer_dict'])




