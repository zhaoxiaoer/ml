import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

learning_rate = 1e-3
batch_size = 64
epochs = 5

# Loading a Dataset
training_data = datasets.FashionMNIST(
    root='../data',
    train=True,
    download=True,
    transform=ToTensor(),
    # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.FashionMNIST(
    root='../data',
    train=False,
    download=False,
    transform=ToTensor()
)

# # Iterating and Visualizing the Dataset
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot"
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
#
# # Preparing your data for training with DataLoaders
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
# # Iterate through the DataLoader
# train_features, train_labels = next(iter(train_dataloader))
# print("Feature batch shape: ", train_features.size())
# print("Labels batch shape: ", train_labels.size())
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap='gray')
# print("Label: ", label)
# plt.title(labels_map[label.item()])
# plt.show()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print("Model structure: ", model, "\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}   [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {(test_loss * 64):>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n--------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
torch.save(model.state_dict(), 'model_weights.pth')

# model.load_state_dict(torch.load('model_weights.pth'))
# model.eval()
# test_loop(test_dataloader, model, loss_fn)
print("111")


# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print("Predicted class: ", y_pred)
#
# input_image = torch.rand(3, 28, 28, device=device)
# print(input_image.size())
# flatten = nn.Flatten()
# flat_image = flatten(input_image)
# print(flat_image.size())
#
# layer1 = nn.Linear(in_features=28*28, out_features=20)
# hidden1 = layer1(flat_image)
# print(hidden1.size())
#
# hidden1 = nn.ReLU()(hidden1)
# print(hidden1.size())
#
# softmax = nn.Softmax(dim=1)
# pred_probab = softmax(hidden1)
# print(pred_probab)
#
# y_pred = pred_probab.argmax(1)
# print(y_pred)


# m = nn.LogSoftmax(dim=1)
# loss = nn.NLLLoss()
# celoss = nn.CrossEntropyLoss()
# inputs = torch.randn(3, 2)
# target = torch.tensor([1, 0, 1])
# out = celoss(inputs, target)
# print(out)

# x = torch.ones(5)
# y = torch.zeros(3)
# w = torch.randn(5, 3, requires_grad=True)
# b = torch.randn(3, requires_grad=True)
# z = torch.matmul(x, w) + b
# loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# print(loss)
# loss.backward()
# print(w.grad)
# print(b.grad)
# print("111")



