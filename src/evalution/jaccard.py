# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 下午2:16
# @Author  : Benqi
"""
Jaccard
"""
import pickle

import numpy as np
from sklearn.metrics import jaccard_similarity_score

from base import Base

class Jaccard(Base):
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_y(self):
        with open(self.dic_config['vector_path'], 'rb') as f:
            self.test_vector_file = pickle.load(f)
            self.test_vector_file = np.array(self.test_vector_file).tolist()
            self.y_true = map(int, self.test_vector_file)


    def load_y_hat(self):
        with open(self.dic_config['predict_path'], 'r') as f:
            self.predicted = f.read()
            self.predicted = self.predicted.split('\n')
            self.y_pred = self.predicted[:-1]

    def evaluate(self):
        self.logging.info('————jaccard距离：————：%s' % (jaccard_similarity_score(self.y_true, self.y_pred)))