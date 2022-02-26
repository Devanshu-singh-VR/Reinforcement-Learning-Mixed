import tensorflow as tf
import numpy as np

a = np.array([1, 2 ,3])
b = tf.convert_to_tensor(a)
print(b.shape)