# -*- coding: utf-8 -*-
# @Time    : 2018/9/3 下午12:11
# @Author  : Benqi

import numpy as np

from base import Base
import util.load_data as load_data

class Bayes(Base):

    """bunch.metrix: like list, every list element is a sample data
       predicted: 1d numpy.array,like a list

    """

    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)

    def load_data(self):
        self.logging.info(self.dic_config['data_path'])
        self.bunch = load_data.read_bunch(self.dic_config['data_path'])
        self.metrix = self.bunch.metrix

    def predict(self):
        self.result = self.model.predict(self.metrix)
        if self.dic_config['format'] == 'proba':
            self.predict_result = self.model.predict_proba(self.metrix)[:, 0]

        else:
            self.predict_result = self.model.predict(self.metrix)

    def dump(self):
        np.savetxt(self.dic_config['predict_path'], self.predict_result, fmt='%s', delimiter=',')
