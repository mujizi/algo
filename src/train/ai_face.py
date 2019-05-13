# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 下午6:33
# @Author  : Benqi
import os
import pickle
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from keras import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from base import Base
from os.path import join
from sklearn.model_selection import train_test_split

class Ai_face(Base):

    """this class is Verification code recognition

       it is a Convolution neural network

       used keras
    """
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)
        self.input_shape = dic_config['input_shape']

    def load_data(self):

        """
        load data from pickle
        :return:
        """

        def standardize(x):

            """standardize function
            """
            return (x - x.mean()) / x.std()

        with open(join(self.data_path), 'rb') as f:
            data_x = pickle.load(f)
            data_y = pickle.load(f)
            reshape_input = list(self.input_shape)[::-1]
            reshape_input.append(-1)
            reshape_input = reshape_input[::-1]

            self.logging.info(reshape_input)
            data_x = data_x.reshape(tuple(reshape_input))
            data_x = standardize(data_x)
            onehot = OneHotEncoder()
            data_y = data_y.reshape((-1,1))
            data_y = onehot.fit_transform(data_y)

            self.logging.info(data_y)
            self.logging.info(data_x.shape)
            self.logging.info(data_y.shape)
            self.data_y = data_y
            self.data_x = data_x

    def train(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))

        model.add(Dense(2, activation='softmax'))

        adm = Adam(lr=0.001, decay=1e-6)

        model.compile(loss=tf.losses.mean_squared_error, optimizer=adm)

        model.fit(self.data_x, self.data_y, batch_size=2, epochs=10,
                  callbacks=[TensorBoard(log_dir=os.path.abspath(os.path.join(self.model_path, '../log')))])

        model.summary()

        self.model = model

    def dump(self):
        self.model.save(self.model_path)