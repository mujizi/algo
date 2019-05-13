# -*- coding: utf-8 -*-
# @Time    : 2018/11/14 下午4:07
# @Author  : Benqi
import os
import numpy as np
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

from base import Base
from util.tools import read_bunch

class Softmax_keras(Base):
    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)

    def load_data(self):
        bunch = read_bunch(self.data_path)

        self.X = np.array(bunch.content)
        self.Y = np.array(bunch.label)

        self.y_true = []
        for i in range(self.Y.shape[0]):
            self.y_true.append(np.argmax(self.Y[i]))


        self.logging.info(self.X.shape)
        self.logging.info(np.array(self.y_true).shape)

    def train(self):
        self.logging.info(self.X[0].shape)

        self.model = keras.Sequential([
            keras.layers.Dense(512, input_dim=1024, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])

        save_dir = os.path.join(os.getcwd())
        checkpoint = ModelCheckpoint(save_dir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callback_list = [checkpoint]

        self.model.fit(self.X, self.y_true, epochs=5)

    def dump(self):
        self.model.save(self.model_path)


    """if you want to load this model, you can use 
    from keras.model import load_model
    
    load_model(model_path)
    """