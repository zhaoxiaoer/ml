import torch
import torch.nn as nn
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

torch.manual_seed(446)
np.random.seed(446)

# we create tensors in a similar way to numpy nd arrays
x_numpy = np.array([0.1, 0.2, 0.3])
x_torch = torch.tensor([0.1, 0.2, 0.3])
print('x_numpy, x_torch')
print(x_numpy, x_torch)
print()

# to and from numpy, pytorch
print('to and from numpy and pytorch')
print(torch.from_numpy(x_numpy), x_torch.numpy())
print()

# we can do basic operation like +-*/
y_numpy = np.array([3, 4, 5])
y_torch = torch.tensor([3, 4, 5])
print('x+y')
print(x_numpy + y_numpy, x_torch + y_torch)
print()

# many functions that are in numpy are also in pytorch
print('norm')
print(np.linalg.norm(x_numpy), torch.norm(x_torch))
print()

# to apply an operation along a dimension,
# we use the dim keyword argument instead of axis
print('mean along the 0th dimension')
x_numpy = np.array([[1, 2], [3, 4.]])
x_torch = torch.tensor([[1, 2], [3, 4.]])
print(np.mean(x_numpy, axis=0), torch.mean(x_torch, dim=0))

# Tensor.view
# "MNIST"
N, C, W, H = 10000, 3, 28, 28
X = torch.randn(N, C, W, H)
print(X.shape)
print(X.view(N, C, 784).shape)
print(X.view(-1, C, 784).shape)

# Broadcasting Semantics
# PyTorch operations support Numpy Broadcasting Semantics
x = torch.empty(5, 1, 4, 1)
y = torch.empty(3, 1, 1)
print((x+y).size())

# Computation graphs
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
c = a + b
d = b + 1
e = c * d
print('c', c)
print('d', d)
print('e', e)

# CUDA
cpu = torch.device('cpu')
gpu = torch.device('cuda')
x = torch.rand(10)
print(x)
x = x.to(gpu)
print(x)
x = x.to(cpu)
print(x)

