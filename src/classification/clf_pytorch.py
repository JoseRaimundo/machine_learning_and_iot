
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
# Acessando os dados
def run_pytorch(x_train, x_test, y_train, y_test, EPOCHS, batch_size):

    x_test, x_valid = train_test_split(x_test, test_size=0.2)
    y_test, y_valid = train_test_split(y_test, test_size=0.2)

    x_train = torch.tensor(x_train)
    y_train = torch.from_numpy(y_train)

    x_valid = torch.from_numpy(x_valid)
    y_valid = torch.tensor(y_valid)

    x_test = torch.tensor(x_test)
    y_test = torch.from_numpy(y_test)

# numerical_data = torch.tensor(numerical_data, dtype=torch.float)
    train_ds     = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size)

    val_ds      = TensorDataset(x_valid, y_valid)
    val_loader  = DataLoader(val_ds, batch_size)

    test_ds     = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_ds, batch_size)
    class SimpleNet(nn.Module):
        # Initialize the layers
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.linear1 = nn.Linear(11, 64)
            self.act1 = nn.ReLU() # Activation function
            self.linear2 = nn.Linear(64, 64)
            self.act2 = nn.ReLU() # Activation function
            # self.linear3 = nn.Linear(64, 64)
            # self.act3 = nn.ReLU() # Activation function
            self.linear4 = nn.Linear(64, 2)
            self.bn4 = nn.BatchNorm1d(2)
            self.act4 = nn.Sigmoid() # Activation function

        # Perform the computation
        def forward(self, x):
            x = self.linear1(x)
            x = self.act1(x)

            x = self.linear2(x)
            x = self.act2(x)

            # x = self.linear3(x)
            # x = self.act3(x)

            x = self.linear4(x)
            x = self.bn4(x)
            x = self.act4(x)
            return x


    model = SimpleNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    def multi_acc(y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)
        return acc

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }


    print("Begin training.")
    for e in (range(EPOCHS)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = loss_fn(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        # VALIDATION    
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')


    with torch.no_grad():
        # for x_batch in test_loader:
        y_pred = model(x_valid)
        # y_pred = np.argmax(y_pred, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        print("Acc: ",accuracy_score(y_pred, y_valid))
        print("AUC: ", roc_auc_score(y_pred, y_valid))

    # y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        print(classification_report(y_valid, y_pred))
