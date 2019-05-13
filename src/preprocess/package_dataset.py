# -*- coding: utf-8 -*-
# @Time    : 2018/11/19 下午2:40
# @Author  : Benqi

from base import Base

import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.datasets import fashion_mnist

class Package_dataset(Base):
    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        sample_nums = x_test.shape[0]
        self.x = x_test.reshape(sample_nums, -1)
        self.y = y_test.reshape(-1, 1)

        one_hot = preprocessing.OneHotEncoder()
        self.y = self.y.tolist()
        self.y = one_hot.fit_transform(self.y)
        self.y = self.y.toarray()

        dataset = np.hstack((self.y, self.x))
        x_list = ['dim' + str(i) for i in range(0, self.x.shape[1])]
        y_list = ['label' + str(i) for i in range(0, self.y.shape[1])]
        self.df = pd.DataFrame(dataset, columns=y_list + x_list)

    def create(self):
        self.df.to_csv(self.dic_config['csv_path'], index=False)
        self.logging.info('————csv to dat successful————')
        self.logging.info('file path: %s' % (self.dic_config['csv_path']))

