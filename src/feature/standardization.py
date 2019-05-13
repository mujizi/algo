# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 下午4:45
# @Author  : Benqi

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from base import Base

class Standardization:
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_data(self):
        self.file = pd.read_csv(self.dic_config['data_path'])
        self.x = self.file.drop([self.dic_config['tag']], axis=1)
        self.x = np.array(self.x)

    def feature_engineer(self):
        sds = StandardScaler()
        self.x_sds = sds.fit_transform(self.x)

    def dump(self):
        np.savetxt(self.dic_config['feature_path'], self.x_sds, fmt='%s', delimiter=',')