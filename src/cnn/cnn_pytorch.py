import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


import torch.utils.data as data
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import numpy as np


def cnn_pytorch(x_train, x_test, y_train, y_test, EPOCHS, batch_size):
    # Carregando dados
    PATH_DATA = '../../databases/cnn/flowers-recognition'
    data_dir = os.path.join(PATH_DATA, 'data')
    train_dir = os.path.join(PATH_DATA, 'train')
    test_dir = os.path.join(PATH_DATA, 'test')

    # Definindo conjuntos de treinamento e test (usando a palavra teste para referi-se à validação)

    data_path = train_dir
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True
    )

    data_path = test_dir
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True
    )


    dataiter = iter(train_loader)

    # Separando labels
    classes = []
    for i in os.listdir(train_dir):
        classes.append(i)

    # Criando rede neural
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            # x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()
    print(type(classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Executando treinamento
    for epoch in range(2):  

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:  
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # Fim do treino