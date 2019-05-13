# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 上午10:57
# @Author  : Benqi

import logging

import pickle

class Base():

    logging = logging

    def __init__(self, dic_config={}):
        self.data_path = dic_config['data_path']
        self.model_path = dic_config['model_path']

    def load_data(self):
        pass

    def train(self):
        pass

    def dump(self):
        with open(self.model_path, 'w') as f:
            pickle.dump(self.model, f)


