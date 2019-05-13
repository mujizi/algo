# -*- coding: utf-8 -*-
# @Time    : 2018/10/29 下午3:23
# @Author  : Benqi
import pickle

from base import Base
import numpy as np

import util.load_data as load_data

class Logistic(Base):
    """

    """
    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)

    # load模型
    def load_model(self):
        with open(self.dic_config['model_path'], 'rb') as f:
            self.params = pickle.load(f)

    def load_data(self):
        self.X, self.y, self.classes = load_data.h5_test_set(self.dic_config['data_path'])

    def predict(self):

        def sigmoid(x):
            y = 1 / (1 + np.exp(-x))
            return y

        w = self.params['w']
        b = self.params['b']
        m = self.X.shape[1]
        Y = np.zeros((1, m))
        w = w.reshape(self.X.shape[0], 1)
        A = sigmoid(np.dot(w.T, self.X) + b)

        for i in range(A.shape[1]):
            if A[0, i] <= 0.5:
                Y[0, i] = 0
            else:
                Y[0, i] = 1

        self.logging.info("test accuracy: {} %".format(100 - np.mean(np.abs(Y - self.y)) * 100))
        self.predict_result = Y

        return Y



