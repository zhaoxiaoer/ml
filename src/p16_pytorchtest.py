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
# x = x.to(gpu)
# print(x)
# x = x.to(cpu)
# print(x)


# PyTorch as an auto grad framework
# demo1
def f(x):
    return (x-2)**2


def fp(x):
    return 2*(x-2)


x = torch.tensor([1.0], requires_grad=True)
y = f(x)
y.backward()
print('Analytical f\'(x):', fp(x))
print('PyTorch\'s f\'(x):', x.grad)


# demo2
def g(w):
    return 2*w[0]*w[1] + w[1]*torch.cos(w[0])


def grad_g(w):
    return torch.tensor([2*w[1] - w[1]*torch.sin(w[0]), 2*w[0] + torch.cos(w[0])])


w = torch.tensor([np.pi, 1], requires_grad=True)
z = g(w)
z.backward()
print('Analytical grad g(w):', grad_g(w))
print('PyTorch\'s grad g(w):', w.grad)

# Using the gradients
x = torch.tensor([5.0], requires_grad=True)
step_size = 0.25
print('iter, \tx, \tf(x), \tf\'(x), \tf\'(x) pytorch')
for i in range(15):
    y = f(x)
    y.backward()  # compute the gradient

    print('{}, \t{:.3f}, \t{:.3f}, \t{:.3f}, \t{:.3f}'.format(i, x.item(), f(x).item(), fp(x).item(), x.grad.item()))

    x.data = x.data - step_size * x.grad  # perform a GD update step

    # We need to zero the grad variable since the backward()
    # call accumulates the gradients in .grad instead of overwriting.
    # The detach_() is for efficiency. You do not need to worry too much about it.
    x.grad.detach_()
    x.grad.zero_()

# Linear Regression
d = 2
n = 50
X = torch.randn(n, d)
true_w = torch.tensor([[-1.0], [2.0]])
y = X @ true_w + torch.randn(n, 1) * 0.1
print('X shape', X.shape)
print('y shape', y.shape)
print('w shape', true_w.shape)


# define a linear model with no bias
def model(X, w):
    return X @ w


# the residual sum of squares loss function
def rss(y, y_hat):
    return torch.norm(y - y_hat)**2 / n


# analytical expression for the gradient
def grad_rss(X, y, w):
    return -2*X.t() @ (y - X @ w) / n


w = torch.tensor([[1.], [0]], requires_grad=True)
y_hat = model(X, w)

loss = rss(y, y_hat)
loss.backward()

print('Analytical gradient', grad_rss(X, y, w).detach().view(2).numpy())
print('PyTorch\'s gradient', w.grad.view(2).numpy())

# Linear regression using GD with automatically computed derivatives
step_size = 0.1
print('iter, \tloss, \tw')
for i in range(20):
    y_hat = model(X, w)
    loss = rss(y, y_hat)

    loss.backward()

    w.data = w.data - step_size * w.grad
    print('{}, \t{:.2f}, \t{}'.format(i, loss.item(), w.view(2).detach().numpy()))

    w.grad.detach()
    w.grad.zero_()

print('\ntrue_w\t\t', true_w.view(2).numpy())
print('estimated w\t', w.view(2).detach().numpy())
print()

# Linear Module
d_in = 3
d_out = 4
linear_module = nn.Linear(d_in, d_out)
example_tensor = torch.tensor([[1., 2, 3], [4, 5, 6]])
# apply a linear transformation to the data
transformed = linear_module(example_tensor)
print('example_tensor', example_tensor.shape)
print('transformed', transformed.shape)
print()
print('We can see that the weights exist in the background\n')
print('W:', linear_module.weight)
print('b:', linear_module.bias)
print()

# Activation functions
activation_fn = nn.ReLU()
example_tensor = torch.tensor([-1.0, 1.0, 0.0])
activated = activation_fn(example_tensor)
print('example_tensor', example_tensor)
print('activated', activated)
print()

# Sequential
d_in = 3
d_hidden = 4
d_out = 1
model = torch.nn.Sequential(
    nn.Linear(d_in, d_hidden),
    nn.Tanh(),
    nn.Linear(d_hidden, d_out),
    nn.Sigmoid()
)
example_tensor = torch.tensor([[1., 2, 3], [4, 5, 6]])
transformed = model(example_tensor)
print('transformed', transformed.shape)
# Access all of the parameters
params = model.parameters()
for param in params:
    print(param)
print()

# Loss functions
mse_loss_fn = nn.MSELoss()
source = torch.tensor([[0., 0, 0]])
target = torch.tensor([[1., 0, -1]])
loss = mse_loss_fn(source, target)
print(loss)
print()

# torch.optim
# create a simple model
model = nn.Linear(1, 1)
# create a simple dataset
X_simple = torch.tensor([[1.]])
y_simple = torch.tensor([[2.]])
# create our optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2)
mse_loss_fn = nn.MSELoss()
y_hat = model(X_simple)
print('model params before:', model.weight)
loss = mse_loss_fn(y_hat, y_simple)
optim.zero_grad()
loss.backward()
optim.step()
print('model params after:', model.weight)
print()

# Linear regression using GD with automatically computed derivatives and PyTorch's Modules
step_size = 0.1
linear_module = nn.Linear(d, 1, bias=False)
loss_func = nn.MSELoss()
optim = torch.optim.SGD(linear_module.parameters(), lr=step_size)
print('iter, \tloss, \tw')
for i in range(200):
    # y_hat = linear_module(X)  # 使用全部数据
    # loss = loss_func(y_hat, y)
    # optim.zero_grad()
    # loss.backward()
    # optim.step()
    # print('{}, \t{:.2f}, \t{}'.format(i, loss.item(), linear_module.weight.view(2).detach().numpy()))

    # 随机梯度下降
    rand_idx = np.random.choice(n)  # take a random point from the dataset
    x = X[rand_idx]
    y_hat = linear_module(x)
    loss = loss_func(y_hat, y[rand_idx])  # only compute the loss on the single point
    optim.zero_grad()
    loss.backward()
    optim.step()
    if i % 20 == 0:
        print('{}, \t{:.2f}, \t{}'.format(i, loss.item(), linear_module.weight.view(2).detach().numpy()))

print('\ntrue w\t\t', true_w.view(2).numpy())
print('estimated w\t', linear_module.weight.view(2).detach().numpy())
print()

# Neural Network Basics in PyTorch
d = 1
n = 200
X = torch.rand(n, 1)
y = 4 * torch.sin(np.pi * X) * torch.cos(6 * np.pi * X**2)
# plt.scatter(X.numpy(), y.numpy())
# plt.title('plot of $f(x)$')
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.show()
# feel free to play with these parameters
step_size = 0.05
momentum = 0.9  # 增加动量后收敛比较快
n_epochs = 6000
n_hidden_1 = 32
n_hidden_2 = 32
d_out = 1
neural_network = nn.Sequential(
    nn.Linear(d, n_hidden_1),
    nn.Tanh(),  # ReLU 就不太好
    nn.Linear(n_hidden_1, n_hidden_2),
    nn.Tanh(),
    nn.Linear(n_hidden_2, d_out)
)
loss_func = nn.MSELoss()
optim = torch.optim.SGD(neural_network.parameters(), lr=step_size, momentum=momentum)
print('iter, \tloss')
for i in range(n_epochs):
    y_hat = neural_network(X)
    loss = loss_func(y_hat, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % (n_epochs // 10) == 0:
        print('{}, \t{:.2f}'.format(i, loss.item()))

X_grid = torch.from_numpy(np.linspace(0, 1, 50)).float().view(-1, d)
y_hat = neural_network(X_grid)
plt.scatter(X.numpy(), y.numpy())
plt.plot(X_grid.detach().numpy(), y_hat.detach().numpy(), c='r')
plt.title('plot of $f(x)$ and $\hat{f}(x)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
print()

# CrossEntropyLoss
loss_func = nn.CrossEntropyLoss()
source = torch.tensor([[-1., 1], [-1., 1], [-1., 1]])
target = torch.tensor([1, 1, 0])
output = loss_func(source, target)
print('output', output)
print()

