import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Carrega os dados
dataframe = pd.read_csv("../../dataset/classification/data/HomeC.csv")

# dataframe = dataframe.drop(axis = 1, labels = ['icon','summary','cloudCover',  'time'])
dataframe = dataframe.drop(axis = 1, labels = ['time','cloudCover'])


def normalize_calss(dataframe = dataframe, target = 'target', limiar = 1):
    class_target = dataframe[target]
    temp_list = []
    for i in class_target.values.tolist():
        if (i > limiar):
            temp_list.append(1)
        else:
            temp_list.append(0)
        
    dataframe = dataframe.drop(axis = 1, labels = [target])
    dataframe['target'] = temp_list
    return dataframe

dataframe = normalize_calss(target='use [kW]')
    
dataframe_temp = dataframe.drop(axis = 1, labels = ['icon','summary'])


categorical_columns = ['icon','summary']
numerical_columns = dataframe_temp.columns.values

outputs = ['target']

for category in categorical_columns:
    dataframe[category] = dataframe[category].astype('category')


# print(dataframe['icon'].head())

icon = dataframe['icon'].cat.codes.values
summary = dataframe['summary'].cat.codes.values

categorical_data = np.stack([icon, summary], 1)
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)


numerical_data = np.stack([dataframe[col].values for col in numerical_columns], 1)
numerical_data = torch.tensor(numerical_data, dtype=torch.float)


outputs = torch.tensor(dataframe[outputs].values).flatten()


categorical_column_sizes = [len(dataframe[column].cat.categories) for column in categorical_columns]
categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes]


total_records = 10000
test_records = int(total_records * .2)

categorical_train_data = categorical_data[:total_records-test_records]
categorical_test_data = categorical_data[total_records-test_records:total_records]
numerical_train_data = numerical_data[:total_records-test_records]
numerical_test_data = numerical_data[total_records-test_records:total_records]
train_outputs = outputs[:total_records-test_records]
test_outputs = outputs[total_records-test_records:total_records]

print(len(categorical_train_data))
print(len(numerical_train_data))
print(len(train_outputs))

print(len(categorical_test_data))
print(len(numerical_test_data))
print(len(test_outputs))


class Model(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)

        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x


model = Model(categorical_embedding_sizes, numerical_data.shape[1], 2, [200,100,50], p=0.4)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 300
aggregated_losses = []

for i in range(epochs):
    i += 1
    y_pred = model(categorical_train_data, numerical_train_data)
    single_loss = loss_function(y_pred, train_outputs)
    aggregated_losses.append(single_loss)

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()

    

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

with torch.no_grad():
    # y_val = model(categorical_test_data, numerical_test_data)
    # loss = loss_function(y_val, test_outputs)

    # with torch.no_grad():
    y_val = model(categorical_test_data, numerical_test_data)
    y_val = np.argmax(y_val, axis=1)
    # loss = loss_function(y_val, test_outputs)
    print(accuracy_score(test_outputs, y_val))
    # print(f'Loss: {loss:.8f}')



