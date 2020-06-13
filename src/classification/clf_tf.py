from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import common

def run_tensorflow(x_train, x_test, y_train, y_test, EPOCHS, BATCH_SIZE):
    train_x, test_x, train_y, test_y = common.laod_traing_test()

    METRICS = [
        keras.metrics.SensitivityAtSpecificity(name='Sen', specificity= 0.5),
        keras.metrics.SpecificityAtSensitivity(name='Spe', sensitivity = 0.5),
        keras.metrics.BinaryAccuracy(name='Acc'),
        keras.metrics.AUC(name='AUC')
    ]

    def make_model(metrics = METRICS, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        model = keras.Sequential([
            keras.layers.Dense( 64, activation='relu'),
            keras.layers.Dense( 64, activation='relu'),
            keras.layers.Dense( 64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )

        return model

    model = make_model(output_bias = 0.00001)


    baseline_history = model.fit(
        train_x,
        train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split = 0.2,
        verbose=1)
    print(">>>>>>>>>>>")
    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y)
    # print(valid_x.shape)
    # print(valid_y.shape)
    print(">>>>>>>>>>>")
    preds = model.predict(test_x, batch_size=BATCH_SIZE)
    baseline_results = model.evaluate(test_x, test_y, batch_size=BATCH_SIZE, verbose=1)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)