# -*- coding: utf-8 -*-
# @Time    : 2018/9/3 下午12:11
# @Author  : Benqi

import util.tools as tools
import util.load_data as load_data

import logging
import numpy as np

class Base():
    logging = logging
    def __init__(self, dic_config={}):
        self.dic_config = dic_config
        self.data_path = dic_config['data_path']
        self.model_path = dic_config['model_path']

    def load_model(self):
        load = tools.Access_model()
        self.model = load.load_model(self.dic_config['model_path'])

    def load_data(self):
        self.bunch = load_data._read_bunch_file(self.dic_config['data_path'])
        self.metrix = self.bunch.metrix

    def predict_batch(self):
        return self.model.predict(self.metrix)

    def predict(self, text=''):
        self.predict_result = self.model.predict(self.metrix)

    def dump(self):
        np.savetxt(self.dic_config['predict_path'], self.predict_result, fmt='%s', delimiter=',')