# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 下午5:05
# @Author  : Benqi
import os

from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense
from matplotlib import pyplot as plt

from base import Base

import pandas as pd

class Ai_lstm(Base):
    """
    """
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)
        self.data_path = dic_config['data_path']
        self.model_path = dic_config['model_path']

    def load_data(self):
        dataset = pd.read_csv(self.data_path, index_col=0)
        dataset.drop(dataset.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
        values = dataset.values

        n_train_hours = 365 * 24
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]
        # split into input and outputs
        self.train_X, self.train_y = train[:, :-1], train[:, -1]
        test_X, self.test_y = test[:, :-1], test[:, -1]

        self.train_X = self.train_X.reshape((self.train_X.shape[0], 1, self.train_X.shape[1]))
        self.test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    def train(self):
        model = Sequential()
        model.add(LSTM(50, input_shape=(self.train_X.shape[1], self.train_X.shape[2]), return_sequences=True))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        # fit train
        history = model.fit(self.train_X,
                            self.train_y,
                            epochs=2,
                            batch_size=72,
                            validation_data=(self.test_X, self.test_y),
                            verbose=2,
                            shuffle=False,
                            callbacks=[TensorBoard(log_dir=os.path.abspath(os.path.join(self.model_path, '../log')))])

        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')

        model.summary()
        self.model = model

    def dump(self):
        self.model.save(self.model_path)

