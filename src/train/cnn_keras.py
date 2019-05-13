# -*- coding: utf-8 -*-
# @Time    : 2018/11/19 下午1:29
# @Author  : Benqi

import logging
import os

from keras import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from base import Base
import util.tools as tools

class Cnn_keras(Base):
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)
        self.lr = dic_config['lr']
        self.epochs = dic_config['epochs']
        self.batch_size = dic_config['batch_size']
        self.input_shape = dic_config['input_shape']
        self.conv_kernel_1 = dic_config['conv_kernel_1']
        self.conv_kernel_2 = dic_config['conv_kernel_2']
        self.channel_1 = dic_config['channel_1']
        self.channel_2 = dic_config['channel_2']
        self.pool_size = dic_config['pool_size']
        self.dense_1 = dic_config['dense_1']
        self.dense_2 = dic_config['dense_2']

    def load_data(self):
        bunch = tools.read_bunch(self.data_path)
        logging.info(bunch.x)
        logging.info(bunch.y)
        self.x = bunch.x
        self.y = bunch.y

        # 保留
        # df = pd.read_csv(self.data_path)
        # df = df.as_matrix()
        # self.x = df[:, 10:]
        # self.y = df[:, 0:10]
        # self.x = self.x.reshape(-1, 28, 28, 1)

    def train(self):
        model = Sequential()

        model.add(Conv2D(self.channel_1, (self.conv_kernel_1), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(self.channel_2, self.conv_kernel_2, activation='relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(self.dense_1, activation='relu'))

        model.add(Dense(self.dense_2, activation='softmax'))

        adm = Adam(lr=self.lr, decay=1e-6)

        model.compile(loss='categorical_crossentropy', optimizer=adm)

        model.fit(self.x, self.y, batch_size=self.batch_size, epochs=self.epochs, callbacks=[TensorBoard(log_dir=os.path.abspath(os.path.join(self.model_path, '../log')))])

        model.summary()

        model.save(self.model_path)

    def dump(self):
        pass



