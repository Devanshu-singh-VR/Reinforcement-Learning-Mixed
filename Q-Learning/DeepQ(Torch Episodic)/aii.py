import torch
import numpy as np

a = torch.zeros((23, 23))
b = a.clone()
print(b)