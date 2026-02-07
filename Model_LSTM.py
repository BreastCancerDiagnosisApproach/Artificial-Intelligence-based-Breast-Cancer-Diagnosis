import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import layers
from Evaluation import evaluation


# Function to build the 3D Adaptive Residual block
def residual_block(x, filters, kernel_size):
    # 3D Convolution with adaptive padding
    x = layers.Conv3D(filters, kernel_size, padding='same')(x)
    # 3D Batch Normalization
    x = layers.BatchNormalization()(x)
    # ReLU activation
    x = layers.Activation('relu')(x)
    return x


def Model_LSTM(train_data, train_target, test_data, test_target):
    out, model = LSTM_train_1(train_data, train_target, test_data)
    pred = np.asarray(out)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, test_target)
    return Eval, pred


def LSTM_train_1(trainX, trainY, testX):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    kernel_size = (3, 3)
    growth_rate = 12
    filters = growth_rate * 2
    x = residual_block(trainX, filters, kernel_size)
    model1 = model.add(Dense(10, activation="relu"))(x)
    model2 = model1.add(Dense(10, activation="relu"))
    model3 = model2.add(Dense(10, activation="relu"))
    model = (model1 + model2 + model3 / 3)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY[:, 1], epochs=2, batch_size=32, verbose=2)
    testPredict = model.predict(testX)
    return testPredict, model
