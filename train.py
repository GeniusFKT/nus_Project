from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datautil import data_loader

n_steps_in, n_steps_out = 5, 1
n_features_in, n_features_out = 27, 1
batch_size = 20


if __name__ == "__main__":
    loader = data_loader.data_loader("D:\\VSCode_Project\\Python\\Project\\data\\level1.csv", n_steps_in, n_steps_out, 0.8)
    trainX, trainY, testX, testY = loader.get_data()

    # define model [Vanilla LSTM]
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features_in)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features_out)))
    model.compile(optimizer='adam', loss='mse')

    model.summary()

    model.fit(trainX, trainY, batch_size=batch_size, epochs=100, validation_data=(testX, testY))
    score = model.evaluate(testX, testY, batch_size=batch_size)
    print('Test score:', score)

    pre = model.predict(testX)

