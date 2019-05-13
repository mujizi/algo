# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 上午10:57
# @Author  : Benqi

import numpy as np
from base import Base

import util.load_data as load_data
import util.tools as tools

class Logistic(Base):

    """lr.optimize: params: X: np.array shape(vector_dim, sample_nums)

                            y: shape(target_nums, sample_nums)

                            w: 1d array shape(vector_dim, 1)

    """

    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)
        self.num_iter = dic_config['num_iter']
        self.learning_rate = dic_config['learning_rate']

    def load_data(self):
        self.x, self.y = load_data.h5_train_set(self.data_path)

    def train(self):
        def sigmoid(x):
            y = 1 / (1 + np.exp(-x))
            return y

        def init_w_b(dim):
            w = np.zeros((dim, 1))
            b = 0
            return w, b

        def propagate(w, b, X, Y):
            """
            X——（num，样本数）
            Y——（类别，样本数）
            cost —— logistic的似然计算出的损失值
            """
            m = X.shape[1]
            A = sigmoid(np.add(np.dot(w.T, X), b))
            cost = -(np.dot(Y, np.log(A).T) + np.dot(1 - Y, np.log(1 - A).T)) / m  # compute cost
            dw = np.dot(X, (A - Y).T) / m
            db = np.sum(A - Y) / m
            cost = np.squeeze(cost)

            grads = {"dw": dw,
                     "db": db}

            return grads, cost

        def optimize(w, b, X, Y, num_iterations, learning_rate):
            costs = []

            for i in range(int(num_iterations)):
                grads, cost = propagate(w, b, X, Y)
                dw = grads["dw"]
                db = grads["db"]
                w = w - learning_rate * dw
                b = b - learning_rate * db
                if i % 100 == 0:
                    costs.append(cost)
                    self.logging.info("Cost after iteration %i: %f" % (i, cost))

            params = {"w": w,
                      "b": b}

            grads = {"dw": dw,
                     "db": db}

            return params, grads, costs

        w, b = init_w_b(self.x.shape[0])
        self.model, grads, costs = optimize(w, b, self.x, self.y, self.num_iter, self.learning_rate)
        return self.model
