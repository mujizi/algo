# -*- coding: utf-8 -*-
# @Time    : 2018/11/19 下午4:28
# @Author  : Benqi

import keras

import numpy as np

from base import Base
import util.tools as tools

class Cnn_keras(Base):

    """bunch.metrix: like list, every list element is a sample data
       predicted: 1d numpy.array,like a list

    """
    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)

    def load_data(self):
        bunch = tools.read_bunch(self.data_path)
        self.logging.info(bunch.x)
        self.logging.info(bunch.y)
        self.x = bunch.x
        self.y = bunch.y

        # dataset = pd.read_csv(self.dic_config['data_path'])
        # dataset = dataset.values
        # self.x = dataset[:, 10:]
        # self.x = self.x.reshape(-1, 28, 28 ,1)
        # self.y = dataset[:, 0:10]

    def load_model(self):
        self.model = keras.models.load_model(self.dic_config['model_path'])

    def predict(self):
        self.predict_result = self.model.predict(self.x)
        self.logging.info(self.predict_result.shape)
        self.logging.info(type(self.predict_result))

        self.pred_list = []
        for i in range(0, self.predict_result.shape[0]):
            self.pred_list.append(np.argmax(self.predict_result[i]))

        self.pred_list = self.pred_list

        a = 0
        self.y_true_list = []
        for i in range(0, self.y.shape[0]):
            self.y_true_list.append(np.argmax(self.y[i]))

            if self.y_true_list[i] == self.pred_list[i]:
                a += 1

        precise = np.true_divide(a, len(self.y_true_list))
        self.logging.info(precise)

        score = self.model.evaluate(self.x, self.y)
        self.logging.info(score)

    def dump(self):
        np.savetxt(self.dic_config['predict_path'], self.pred_list, fmt='%s', delimiter=',')