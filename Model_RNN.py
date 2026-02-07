import numpy as np
from keras.layers import LSTM, Dense
import tensorflow as tf
from keras.models import Sequential
from Evaluation import evaluation


def Model_RNN(train_data, train_target, test_data, test_target):
    out, model = RNN_train(train_data, train_target, test_data)
    out = np.reshape(out, test_target.shape)
    pred = np.round(out)
    Eval = evaluation(pred, test_target)
    return Eval, pred


def RNN_train(trainX, trainY, testX):
    trainX = np.reshape(trainX, (trainX.shape[0],  trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0],  testX.shape[1]))
    model = Sequential()
    model.add(Dense(trainY.shape[1]))
    model.add(tf.random.uniform(trainX, minval=0.01, maxval=0.99),dtype='dtype')
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=200, batch_size=64, verbose=2)
    testPredict = model.predict(testX)
    return testPredict, model

