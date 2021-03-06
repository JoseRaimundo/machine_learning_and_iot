import datetime
import common
import main_pytorch as pt
import main_tf as tf

option = int(input('Informe lib (1): Pytorch - (2):  Tensorflow: '))

x_train, x_test, y_train, y_test = common.laod_traing_test()
batch_size = common.getBatchSize()
EPOCHS = common.getEpochs()

first_time = datetime.datetime.now()
if (option == 1):
    pt.run_pytorch(x_train, x_test, y_train, y_test, EPOCHS, batch_size)
else:
    tf.run_tensorflow(x_train, x_test, y_train, y_test, EPOCHS, batch_size)

second_time = datetime.datetime.now()
print("Total time: " , (second_time - first_time))
