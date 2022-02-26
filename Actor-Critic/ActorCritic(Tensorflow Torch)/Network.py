from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

class ACNetwork(Model):
    def __init__(self, num_actions):
        super(ACNetwork, self).__init__()
        self.num_actions = num_actions
        self.fc1 = Dense(1024, activation='relu')
        self.fc2 = Dense(512, activation='relu')
        self.softmax = Dense(num_actions, activation='softmax')
        self.value = Dense(1, activation='linear')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        policy = self.softmax(x)
        value = self.value(x)
        return policy, value

if __name__ == '__main__':
    network = ACNetwork(4)
    test = tf.ones((1, 4))
    policy, value = network(test)
    policy = policy.numpy()
    print(policy.shape)