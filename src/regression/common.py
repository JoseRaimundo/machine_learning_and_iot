import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = '../../databases/smart_house/data/HomeC.csv'
EPOCHS = 100
BATCH_SIZE = 100

def getBatchSize():
    return BATCH_SIZE

def getEpochs():
    return EPOCHS


def convert_categorical(dataframe, labels):
    for label in labels:
        if(type(dataframe[label][0]) is str):
            le = LabelEncoder()
            le.fit(dataframe[label])
            dataframe[label] =  le.transform(dataframe[label])
    return dataframe


def laod_traing_test(path = DATA_PATH, target = 'use [kW]'):
    print("... Acessando data .... ")
    df = pd.read_csv(path)
    df = df[:10000]

    denecessary_features = [
        'cloudCover', 
        ]

    print("... Removendo features desnecessáris .... ")
    df = df.drop(axis = 1, labels = denecessary_features)

    print("... Consertando/removendo campos nulos .... ")
    df = df.dropna()
    # print(df.index)
    # print(df.dtypes)
    print("... Encodando features categóricas ...." )
    convert_categorical(df, df.columns.values)

    print("... Dividindo dataset em treino e teste ... ")

    y = df[target]
    X = df.drop(axis = 1, labels = [target])

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print("... Normalizando dados .... ")
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test  = scaler.transform(x_test)
    print("... Retornando: x_train, x_test, y_train, y_test .... ")
    return x_train.astype('float32'), x_test.astype('float32'), y_train.to_numpy().astype('float32'), y_test.to_numpy().astype('float32')

