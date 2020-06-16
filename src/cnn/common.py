from tensorflow.examples.tutorials.mnist import input_data


BATCH_SIZE = 30
EPOCHS = 1000
DATASET_PATH = 'data/fashion'

def getBatchSize():
    return BATCH_SIZE

def getEpochs():
    return EPOCHS

data = input_data.read_data_sets(DATASET_PATH)
data.train.next_batch(BATCH_SIZE)

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def laod_traing_test():
    X_train, y_train = load_mnist(DATASET_PATH, kind='train')
    X_test, y_test = load_mnist(DATASET_PATH, kind='t10k')
    X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
    X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))

    return X_train, X_test, y_train, y_test


# a, b, c, d = load_traing_test()
# print(d)
# print(d.shape)