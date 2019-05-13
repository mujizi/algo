# -*- coding: utf-8 -*-
# @Time    : 2018/11/7 上午10:29
# @Author  : Benqi
"""
input: csv file
output: used pickle to save .dat type file
"""
from base import Base

import pandas as pd
import pickle

class Csv2dat(Base):

    """from csv to .dat
            pickle_path: can use pickle.load(f) open

    """

    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)

    def load_data(self):
        self.file = pd.read_csv(self.dic_config['raw_data'])
        self.x = self.file.drop([self.dic_config['tag']], axis=1)

    def create(self):
        with open(self.dic_config['pickle_path'], 'wb') as f:
            pickle.dump(self.x, f)

        self.logging.info('————csv to dat successful————')
        self.logging.info('file path: %s' % (self.dic_config['pickle_path']))

