import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
from Network import ACNetwork
import numpy as np

class Agent(object):
    def __init__(self, alpha, gamma, num_actions):
        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions
        self.action_space = np.array([i for i in range(num_actions)])

        # Calling the ACNetwork
        self.network = ACNetwork(num_actions)
        self.optimizer = keras.optimizers.Adam(learning_rate=alpha)

    def choose_action(self, state):
        state = tf.expand_dims(
            tf.convert_to_tensor(state), axis=0
        )
        policy, value = self.network(state)
        policy = policy.numpy()

        # take out the action
        action = np.random.choice(self.action_space, p=policy[0])
        return action

    def learn(self, state, action, reward, next_state, done):
        state = tf.convert_to_tensor([state])
        next_state = tf.convert_to_tensor([next_state])
        reward = tf.convert_to_tensor([reward])

        with tf.GradientTape() as tape:
            policy, state_value = self.network(state)
            _, next_state_value = self.network(next_state)

            log_lik = K.log(policy[0][action]) # ln(pi(s/a))
            delta = reward + self.gamma*next_state_value*(1-int(done)) - state_value

            # ACLoss
            actor_loss = -log_lik*delta
            critic_loss = delta**2
            ACLoss = actor_loss + critic_loss

        gradient = tape.gradient(ACLoss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.network.trainable_variables))

    def load_checkpoint(self, path):
        self.network.load_weights(path)

    def save_checkpoint(self, path):
        self.network.save_weights(path)
