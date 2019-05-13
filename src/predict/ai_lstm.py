# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 下午7:14
# @Author  : Benqi
import keras
import numpy as np
import pandas as pd
from base import Base

class Ai_lstm(Base):
    """
    """
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)
        self.data_path = dic_config['data_path']
        self.model_path =dic_config['model_path']
        self.predict_path = dic_config['predict_path']

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

    def load_model(self):
        self.model = keras.models.load_model(self.model_path)

    def predict(self):
        self.pred = self.model.predict(self.test_X)
        self.pred = np.reshape(self.pred, (-1, 1))
        self.test_y = np.reshape(self.test_y, (-1, 1))

    def dump(self):
        data_y = np.hstack((self.pred, self.test_y))
        df = pd.DataFrame(data_y, columns=['pred', 'y_true'])

        df.to_csv(self.predict_path)

