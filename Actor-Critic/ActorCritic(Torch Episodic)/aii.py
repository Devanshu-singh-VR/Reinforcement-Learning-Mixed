from tensorflow.keras import backend as K
import tensorflow as tf
import torch

print(torch.log(torch.tensor(3)))
print(K.log(float(3.0)))

print(torch.square(torch.tensor(2)))