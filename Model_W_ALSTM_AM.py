import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.layers import *
from keras.layers import LSTM

from Evaluation import evaluat_error


class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences

        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1))
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1))
        super(attention,self).build(input_shape)


def call(self, x):
    e = K.tanh(K.dot(x,self.W)+self.b)
    a = K.softmax(e, axis=1)
    output = x*a
    if self.return_sequences:

        return output
    return K.sum(output, axis=1)

# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def LSTM_train(trainX, trainY, testX, sol):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(int(sol[0]), input_shape=(1, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    testPredict = np.zeros((testX.shape[0], trainY.shape[1])).astype('int')
    for i in range(trainY.shape[1]):
        print(i)
        model.add_update(attention)
        model.fit(trainX, trainY[:, i].reshape(-1, 1), epochs=round(sol[1]), batch_size=1, verbose=2)
        testPredict[:, i] = model.predict(testX).ravel()
    return testPredict, model

def Model_W_ALSTM_AM(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [5,5]
    out, model = LSTM_train(train_data, train_target, test_data, sol)
    Eval = evaluat_error(out, test_target)
    return np.asarray(Eval)[:,0]


