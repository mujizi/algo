# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 下午2:58
# @Author  : Benqi
"""
cofuse matrix
"""
from base import Base

import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import util.load_data as load_data

class Confuse_matrix(Base):

    """params: y: sample true label, 1d array or list[]
               y_hat: predicted data saved in directory predict，1d array or list[]
               labels: sample label_list

    """

    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)

    def load_y(self):
        self.bunch = load_data.read_bunch(self.dic_config['vector_path'])
        self.y = self.bunch.label


    def load_y_hat(self):
        self.y_hat = load_data.read_file(self.dic_config['predict_path'])
        self.y_hat = self.y_hat.split('\n')[:-1]


    def evaluate(self):
        self.load_y()
        self.load_y_hat()
        label_list = os.listdir(self.dic_config['raw_data'])

        mat = confusion_matrix(self.y, self.y_hat, labels=label_list)
        sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=True,
                    xticklabels=False,
                    yticklabels=False)
        plt.title('confusion matrix')
        plt.savefig(self.dic_config['image_path'], format='png')
        plt.show()
