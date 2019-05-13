# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 下午4:41
# @Author  : Benqi
"""
F1
"""
import pickle
import numpy as np
from sklearn import metrics

from base import Base

# 准确率，召回率，F1
class F1(Base):

    """f1, precision_score, recall_score
            params: y_ture :1d array or sparse matrix
                    y_hat : 1d array or sparse matrix
    """

    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)

    def load_y(self):
        with open(self.dic_config['vector_path'], 'rb') as f:
            self.test_vector_file = pickle.load(f)
            self.test_vector_file = np.array(self.test_vector_file.label).tolist()
            # self.test_vector_file = map(int, self.test_vector_file)

    def load_y_hat(self):
        with open(self.dic_config['predict_path'], 'r') as f:
            self.predicted = f.read()
        self.predicted = self.predicted.split('\n')
        self.predicted = self.predicted[:-1]
        self.logging.info(type(self.predicted[0]))
        # self.predicted = map(int, self.predicted)

    def evaluate(self):
        self.logging.info('精度:{0:.3f}'.format(metrics.precision_score(self.test_vector_file, self.predicted, average='weighted')))
        self.logging.info('召回:{0:0.3f}'.format(metrics.recall_score(self.test_vector_file, self.predicted, average='weighted')))
        self.logging.info('f1-score:{0:.3f}'.format(metrics.f1_score(self.test_vector_file, self.predicted, average='weighted')))



        # 保留
        # for tlabel, file_name, exp_cate in zip(self.test_vector_file, self.test_vector_file.filenames, self.predicted):
        #     if tlabel != exp_cate:
        #         logging.info("实际," + tlabel + "————预测————" + exp_cate)