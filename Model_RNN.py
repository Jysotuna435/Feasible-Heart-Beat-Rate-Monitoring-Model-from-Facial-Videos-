import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense

from Evaluation import evaluat_error


def RNN_train(trainX, trainY, testX):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(200, input_shape=(1, trainX.shape[2])))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
    testPredict = model.predict(testX)
    return testPredict, model

def Model_RNN(train_data, train_target, test_data, test_target):
    out, model = RNN_train(train_data, train_target, test_data)  # RNN
    out = np.reshape(out, test_target.shape)
    pred = np.round(out)
    Eval = evaluat_error(pred.reshape(-1,1), test_target.reshape(-1,1))
    return np.asarray(Eval).ravel(),pred

