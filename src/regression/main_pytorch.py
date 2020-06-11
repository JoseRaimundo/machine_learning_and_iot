import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Acessando os dados
def run_pytorch(x_train, x_test, y_train, y_test, EPOCHS, batch_size):
    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    train_ds =  TensorDataset(x_train, y_train)


    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    # # MSE loss
    def mse(t1, t2):
        diff = t1 - t2
        return torch.sum(diff * diff) / diff.numel()

    class SimpleNet(nn.Module):
        # Initialize the layers
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.linear1 = nn.Linear(30, 64)
            self.act1 = nn.ReLU() # Activation function
            self.linear2 = nn.Linear(64, 64)
            self.linear3 = nn.Linear(64, 1)

        # Perform the computation
        def forward(self, x):
            x = self.linear1(x)
            x = self.act1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            return x


    model = SimpleNet()

    criterion = torch.nn.MSELoss() 
    opt = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.0001)
    loss_fn = F.mse_loss


    def fit(loss_fn, opt):
        for epoch in range(EPOCHS):
            for xb,yb in train_dl:
                pred = model(Variable(xb))
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()
                print('epoch {}, loss {}'.format(epoch, loss.item()))

    fit( loss_fn, opt)
    preds = model(Variable(x_test))

    for i in range(10):
        print({y_test[i], " " , preds[i]})

    print("|------------- RESULTADO --------------- |")
    print("|- MSE : ", mse(preds, y_test).item()/10)
    print("|------------- RESULTADO --------------- |")