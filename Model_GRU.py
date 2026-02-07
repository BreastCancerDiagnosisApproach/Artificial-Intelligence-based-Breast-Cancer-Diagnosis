# Importing the libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Evaluation import evaluation
plt.style.use('fivethirtyeight')
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from sklearn.linear_model import RidgeClassifier


def Model_GRU(X_train, y_train, X_test, Y_test):
    # The GRU architecture
    regressorGRU = Sequential()
    # First GRU layer with Dropout regularisation
    regressorGRU.add(
        GRU(units=10, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Second GRU layer
    regressorGRU.add(GRU(units=10, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Third GRU layer
    regressorGRU.add(GRU(units=10, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Fourth GRU layer
    regressorGRU.add(GRU(units=10, activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # The output layer
    regressorGRU.add(Dense(10))  # units=1
    # Compiling the RNN
    regressorGRU.compile(optimizer='Adam', loss='mean_squared_error')
    regressorGRU.add(tf.random.uniform(X_train, minval=0.01, maxval=0.99),
                     dtype='dtype')
    # Fitting to the training set
    trx_data = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    tex_data = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    pred = np.zeros((Y_test.shape))
    for i in range(y_train.shape[1]):  # for all classes
        regressorGRU.fit(trx_data, y_train[:, i], epochs=2, batch_size=64)
        pr = regressorGRU.predict(tex_data).ravel()
        for j in range(pr.shape[0]):
            pred[j, i] = np.mean(pr[j])
            if np.isnan(pred[j, i]):
                pred[j, i] = 0
        RidgeClassifier().fit(pred, Y_test)
    Eval = evaluation(pred, Y_test)
    return Eval, pred
