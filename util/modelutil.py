import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from util import data_loader
from seq2seq.models import Seq2Seq
from keras.callbacks import ModelCheckpoint

n_steps_in, n_steps_out = 5, 1
n_features_in, n_features_out = 27, 1
batch_size = 20
EPOCHS = 30

class my_model():
    def __init__(self):
        self.loader = data_loader.data_loader('/home/zhangbowen/zbw/nus_Project/data/level1.csv', n_steps_in, n_steps_out, 0.8)
        trainX, trainY, testX, testY = self.loader.get_data()
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

class model_origin(my_model):
    def train(self):
        model = Sequential()
        model.add(LSTM(128, activation='relu', input_shape=(n_steps_in, n_features_in), return_sequences=True))
        model.add(LSTM(256, activation='relu', input_shape=(n_steps_in, 128), return_sequences=True))
        model.add(LSTM(512, activation='relu', input_shape=(n_steps_in, 256), return_sequences=False))

        model.add(RepeatVector(n_steps_out))
        model.add(LSTM(512, activation='relu', input_shape=(n_steps_out, 512), return_sequences=True))
        model.add(LSTM(256, activation='relu', input_shape=(n_steps_out, 512), return_sequences=True))
        model.add(LSTM(128, activation='relu', input_shape=(n_steps_out, 256), return_sequences=True))

        model.add(TimeDistributed(Dense(n_features_out)))

        loss = tf.keras.losses.MeanSquaredError(name="Loss")

        metrics = [tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage"),
                   tf.keras.metrics.MeanSquaredError(name="mean_square")]

        checkpoint = ModelCheckpoint("model_save", monitor="mean_absolute_percentage", verbose=1, save_best_only=True, mode='min')

        model.compile(optimizer='adam', loss=loss, metrics=metrics)

        model.summary()

        model.fit(self.trainX, self.trainY, batch_size=batch_size, callbacks=[checkpoint], epochs=EPOCHS, validation_data=(self.testX, self.testY), shuffle=True)


class model_seq2seq(my_model):
    def train(self):
        model = Seq2Seq(batch_input_shape=(batch_size, n_steps_in, n_features_in), hidden_dim=200, output_length=n_steps_out, output_dim=n_features_out, depth=3)

        loss = tf.keras.losses.MeanSquaredError(name="Loss")

        metrics = [tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage"),
                   tf.keras.metrics.MeanSquaredError(name="mean_square")]

        checkpoint = ModelCheckpoint("model_save", monitor="mean_absolute_percentage", verbose=1, save_best_only=True, mode='min')

        model.compile(optimizer='adam', loss=loss, metrics=metrics)

        model.summary()

        model.fit(self.trainX, self.trainY, batch_size=batch_size, callbacks=[checkpoint], epochs=EPOCHS, validation_data=(self.testX, self.testY), shuffle=True)
