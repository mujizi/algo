# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 下午7:25
# @Author  : Benqi
import sys
sys.path.append('../')

import numpy as np
from tensorflow import keras

import util.load_data as load_data

class Softmax_keras():
    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)

    def load_data(self, data_path):
        bunch = load_data.read_bunch(data_path)
        x_test = bunch.content
        y_true = bunch.label
        print(y_true.shape)
        return x_test, y_true

    def predict(self, data_path):
        self.x_test, self.y_true = self.load_data(data_path)
        self.predicted = self.model.predict(self.x_test)

        self.model.summary()
        print(self.predicted.shape)

        a = 0
        self.pred_list = []
        self.y_true_list = []
        for i in range(0, self.predicted.shape[0]):
            self.pred_list.append(np.argmax(self.predicted[i]))
            self.y_true_list.append(np.argmax(self.y_true[i]))
        for i in range(0, self.predicted.shape[0]):
            if self.pred_list[i] == self.y_true_list[i]:
                a += 1

        print(np.true_divide(a, self.predicted.shape[0]))

if __name__ == '__main__':
    soft_max = Softmax_keras()
    soft_max.load_model('/Users/ouhon/PycharmProjects/ai_algo/data/softmax_keras/train/softmax_keras.h5')
    soft_max.predict('/Users/ouhon/PycharmProjects/ai_algo/data/softmax_keras/preprocess/test_digits.dat')
