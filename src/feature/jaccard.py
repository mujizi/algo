# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 下午4:25
# @Author  : Benqi
import logging

import util.load_data as ld

from src.evalution.base import Base
from sklearn.metrics import jaccard_similarity_score

class Jaccard(Base):

    '''param: y_true and y_pred is 1d array or matirx
    '''

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_data(self):
        with open(self.dic_config['predict_path'], 'r') as f:
            self.predicted = f.read()
        self.predicted = self.predicted.split('\n')
        self.y_pred = self.predicted[:-1]

    def feature_engineer(self):
        self.logging.info('————jaccard距离：————：%s' % (jaccard_similarity_score(self.y_true, self.y_pred)))