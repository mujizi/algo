# -*- coding: utf-8 -*-
# @Time    : 2018/12/10 下午3:45
# @Author  : Benqi
from base import Base
import pandas as pd
from scipy import sparse

class Matrix2sparse(Base):
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)
        self.data_path = dic_config['data_path']
        self.sparse_path = dic_config['sparse_path']

    def load_data(self):
        dataset = pd.read_csv(self.data_path)
        self.y_train = dataset['label'].values
        self.x_train = dataset.drop(['label'], axis=1).values

    def create(self):
        self.x_feature = sparse.csc_matrix(self.x_train)
        sparse.save_npz(self.sparse_path, self.x_feature)
        self.logging.info('————change to sparse matrix successful————')
        self.logging.info('file path: %s' % (self.sparse_path))