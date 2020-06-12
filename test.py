# Import Numpy & PyTorch
import numpy as np
import torch
import torch.nn as nn

# Create tensors.
# x = torch.tensor(3.)
# w = torch.tensor(4., requires_grad=True)
# b = torch.tensor(5., requires_grad=True)
# # Print tensors
# print(x)
# print(w)

# # Arithmetic operations
# y = w * x + b
# print(y)
# # Compute gradients
# y.backward()

# # Input (temp, rainfall, humidity)
# inputs = np.array([[73, 67, 43], 
#                    [91, 88, 64], 
#                    [87, 134, 58], 
#                    [102, 43, 37], 
#                    [69, 96, 70]], dtype='float32')

# # Targets (apples, oranges)
# targets = np.array([[56, 70], 
#                     [81, 101], 
#                     [119, 133], 
#                     [22, 37], 
#                     [103, 119]], dtype='float32')


# # Convert inputs and targets to tensors
# inputs = torch.from_numpy(inputs)
# targets = torch.from_numpy(targets)
# print(inputs)
# print(targets)


# # Weights and biases
# w = torch.randn(2, 3, requires_grad=True)
# b = torch.randn(2, requires_grad=True)
# print(w)
# print(b)

# # Define the model
# def model(x):
#     return x @ w.t() + b


# # Generate predictions
# preds = model(inputs)
# print(preds)


# # MSE loss
# def mse(t1, t2):
#     diff = t1 - t2
#     return torch.sum(diff * diff) / diff.numel()

# # Compute loss
# loss = mse(preds, targets)
# print(loss)


# # Compute gradients
# loss.backward()

# # Gradients for weights
# print(w)
# print(w.grad)

# # Gradients for bias
# print(b)
# print(b.grad)
# preds = model(inputs)
# print(preds)


# # Calculate the loss
# loss = mse(preds, targets)
# print(loss)

# print(b)

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119], 
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119], 
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')


inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

from torch.utils.data import TensorDataset, DataLoader


# Define dataset
train_ds = TensorDataset(inputs, targets)
print(train_ds[0:3])
print()

# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
print(next(iter(train_dl)))


# Define model
model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)


# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Import nn.functional
import torch.nn.functional as F

# Define loss function
loss_fn = F.mse_loss


loss = loss_fn(model(inputs), targets)
print(loss)


# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
    print('Training loss: ', loss_fn(model(inputs), targets))

fit(100, model, loss_fn, opt)

# Generate predictions
preds = model(inputs)
print(preds)
print(targets)

class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 3)
        self.act1 = nn.ReLU() # Activation function
        self.linear2 = nn.Linear(3, 2)
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x


model = SimpleNet()
opt = torch.optim.SGD(model.parameters(), 1e-5)
loss_fn = F.mse_loss

fit(100, model, loss_fn, opt)





















