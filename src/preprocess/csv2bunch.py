# -*- coding: utf-8 -*-
# @Time    : 2018/12/3 上午11:41
# @Author  : Benqi
"""
input: bunch file,bunch like a dictionary format
output: csv file
"""
from sklearn.utils import Bunch
from util.load_data import write_bunch
from base import Base

import pandas as pd

class Csv2bunch(Base):

    """
    """

    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)
        self.csv_path = dic_config['csv_path']
        self.bunch_path = dic_config['bunch_path']

    def load_data(self):
        self.content = pd.read_csv(self.csv_path)

    def create(self):
        self.label = self.content['label'].values
        self.feature = self.content.drop(['label'], axis=1).values

        bunch = Bunch()
        bunch.label = self.label
        bunch.feature = self.feature
        write_bunch(self.bunch_path, bunch)

        self.logging.info(self.content)
        self.logging.info(type(self.content['label']))

        self.logging.info('————csv to bunch successful————')
        self.logging.info('file path: %s' % (self.dic_config['bunch_path']))

