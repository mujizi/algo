# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 下午7:15
# @Author  : Benqi
import sys
sys.path.append('../')

import pickle
import numpy as np
import util.load_data as load_data

class Logistic():
    # load模型
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.params = pickle.load(f)

    def load_data(self, data_path):
        X, y, classes = load_data.h5_test_set(data_path)

        return X, y, classes

    def predict(self, data_path):
        X, y, classes = self.load_data(data_path)

        def sigmoid(x):
            y = 1 / (1 + np.exp(-x))
            return y

        w = self.params['w']
        b = self.params['b']
        m = X.shape[1]
        Y = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)
        A = sigmoid(np.dot(w.T, X) + b)

        for i in range(A.shape[1]):
            if A[0, i] <= 0.5:
                Y[0, i] = 0
            else:
                Y[0, i] = 1

        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y - y)) * 100))
        self.predict_result = Y

        return Y

if __name__ == '__main__':
    logistic = Logistic()
    logistic.load_model('/Users/ouhon/PycharmProjects/ai_algo/data/lr/train/logistic.m')
    logistic.predict('/Users/ouhon/PycharmProjects/ai_algo/data/lr/raw/test_catvnoncat.h5')

