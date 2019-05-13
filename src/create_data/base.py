# -*- coding: utf-8 -*-
# @Time    : 2018/11/29 下午6:50
# @Author  : Benqi

import logging

class Base:
    logging = logging

    def __init__(self, dic_config={}):
        self.dic_config = dic_config

    def create(self):
        pass