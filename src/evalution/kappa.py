# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 下午4:24
# @Author  : Benqi
"""
kappa
"""

from sklearn.metrics import cohen_kappa_score

from base import Base
import util.load_data as load_data

class Kappa(Base):

    '''param: y_true and y_pred is 1d array or matirx
    '''

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_y(self):
        self.bunch = load_data._read_bunch_file(self.dic_config['vector_path'])
        self.y_true = self.bunch.label

    def load_y_hat(self):
        with open(self.dic_config['predict_path'], 'r') as f:
            self.predicted = f.read()
        self.predicted = self.predicted.split('\n')
        self.y_pred = self.predicted[:-1]

    def evaluate(self):
        self.logging.info('————kappa系数值————：%s' % (cohen_kappa_score(self.y_true, self.y_pred)))