# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 下午2:58
# @Author  : Benqi
import logging
import util.tools as to

class Base():
    logging = logging
    def __init__(self, dic_config={}):
        self.dic_config = dic_config

    def load_y(self):
        self.y = to._read_bunch_file(self.dic_config['vector_path'])

    def load_y_hat(self):
        self.y_hat = to.read_file_encode(self.dic_config['predict_path', 'utf-8'])

    def evaluate(self):
        pass