# -*- coding: utf-8 -*-
# @Time    : 2018/11/6 下午4:54
# @Author  : Benqi
"""
h5py is a dataset file ,format like a dictionary
output: bunch file
"""
import h5py
import pickle
import numpy as np

from base import Base
from sklearn.utils import Bunch

class H5py2bunch(Base):

    """from h5py file to bunch
            params: raw_data: h5py
                    bunch_path: .dat file
                    Bunch(x=[], label=[]),lable: true label
                                          x: feature data

    """

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_data(self):
        with h5py.File(self.dic_config['raw_data'], 'r') as self.f:
            self.data_x = np.array(self.f[self.dic_config['x_tag']][:])
            self.data_y = np.array(self.f[self.dic_config['y_tag']][:])

    def create(self):
        self.bunch = Bunch(x=[], label=[])
        self.bunch.x.append(self.data_x)
        self.bunch.label.append(self.data_y)


        with open(self.dic_config['bunch_path'], 'wb') as f:
            pickle.dump(self.bunch, f, protocol=2)
        self.logging.info('————h5py to bunch successful————')
        self.logging.info('file path：%s' % (self.dic_config['bunch_path']))