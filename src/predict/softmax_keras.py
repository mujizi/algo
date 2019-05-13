# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 下午4:26
# @Author  : Benqi
import numpy as np
from tensorflow import keras

from base import Base
import util.load_data as load_data

class Softmax_keras(Base):
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_model(self):
        self.model = keras.models.load_model(self.dic_config['model_path'])

        # bunch = load_data.read_bunch(self.dic_config['data_path'])
        #
        # self.x_test = bunch.content
        # self.y_true = bunch.label
        # self.y_true = np.squeeze(self.y_true).tolist()
        # logging.info(self.y_true)
        # self.y_true = map(int, self.y_true)

    def load_data(self):
        bunch = load_data.read_bunch(self.dic_config['data_path'])
        self.x_test = bunch.content
        self.y_true = bunch.label
        self.logging.info(self.y_true.shape)

    def predict(self):
        self.predicted = self.model.predict(self.x_test)

        self.model.summary()
        self.logging.info(self.predicted.shape)

        a = 0
        self.pred_list = []
        self.y_true_list = []
        for i in range(0, self.predicted.shape[0]):
            self.pred_list.append(np.argmax(self.predicted[i]))
            self.y_true_list.append(np.argmax(self.y_true[i]))
        for i in range(0, self.predicted.shape[0]):
            if self.pred_list[i] == self.y_true_list[i]:
                a += 1

        self.logging.info(np.true_divide(a, self.predicted.shape[0]))

    def dump(self):
        """np.savetxt is used to save 1d numpy array or list format data.
        """
        np.savetxt(self.dic_config['predict_path'], self.pred_list, fmt='%s', delimiter=',')
        self.logging.info('————model restore successful————')
