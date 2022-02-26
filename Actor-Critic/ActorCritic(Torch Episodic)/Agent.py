import torch
import torch.optim as optim
import numpy as np
from Network import ACNetwork

class ACAgent(object):
    def __init__(self, alpha, gamma,
                 num_actions, num_states):
        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions
        self.num_states = num_states
        self.network = ACNetwork(num_actions, num_states)
        self.action_space = np.array([i for i in range(num_actions)])
        self.optimizer = optim.Adam(self.network.parameters(), lr=alpha)

    def choose_action(self, state):
        state = torch.tensor(state).unsqueeze(0)
        pi, value = self.network(state)
        pi = pi.squeeze(0).detach().numpy()
        action = np.random.choice(self.action_space, p=pi)
        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor(state).unsqueeze(0)
        next_state = torch.tensor(next_state).unsqueeze(0)
        reward = torch.tensor(reward)

        pi, value = self.network(state)
        pi_, next_value = self.network(next_state)

        # Delta cal
        target = reward + self.gamma*next_value*(1 - int(done))
        delta = target - value

        # Log of policy
        log_like = torch.log(pi[0][action])

        # Total Loss
        critic_loss = torch.square(delta)
        actor_loss = -delta*log_like
        total_loss = critic_loss+actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def save_checkpoint(self, state, path, epoch):
        print('save the checkpoint ', epoch)
        torch.save(state, path)

    def load_checkpoint(self, saved_checkpoint):
        print('loading the checkpoint')
        self.network.load_state_dict((saved_checkpoint['model_dict']))
        self.optimizer.load_state_dict(saved_checkpoint['optimizer_dict'])
