# -*- coding: utf-8 -*-
# @Time    : 2018/9/3 下午12:11
# @Author  : Benqi
"""
two classification problem assessment
"""

# 科学计算
import numpy as np
import matplotlib.pyplot as plt

# sklearn工具
from sklearn import metrics
from sklearn.metrics import roc_curve

from base import Base
import util.tools as tools

class Roc(Base):

    """this function used to describe two classification problem
            param: y: true label,1d array list
                   Y_hat: predicted is positive probability，1d array or list
                   target_name[0]: set positive cases
    """

    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)

    def load_y(self):
        bunch = tools._read_bunch_file(self.dic_config['vector_path'])
        self.y = bunch.label
        self.logging.info(type(self.y))
        self.logging.info(len(self.y))
        self.target_name = bunch.target_name

    def load_y_hat(self):
        self.y_hat = np.loadtxt(self.dic_config['predict_path'])
        self.logging.info(type(self.y_hat))
        self.logging.info(self.y_hat)

    def evaluate(self):
        self.logging.info(type(self.y_hat))
        fpr, tpr, thresholds = roc_curve(self.y, self.y_hat, pos_label=self.target_name[0])
        auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.title('ROC_curve' + '(AUC: ' + str(auc) + ')')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(self.dic_config['image_path'])
        self.logging.info("AUC的值为：{}".format(auc))
        plt.show()