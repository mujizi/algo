# -*- coding: utf-8 -*-
# @Time    : 2018/10/15 下午3:24
# @Author  : Benqi
import logging

class Base:
    logging = logging

    def __init__(self, dic_config={}):
        self.dic_config = dic_config

    def load_data(self):
        pass

    def create(self):
        pass