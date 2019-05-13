# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 下午3:17
# @Author  : Benqi
"""
txt to bunch file
"""
import pickle
import numpy as np
from sklearn import preprocessing

from sklearn.utils import Bunch

from base import Base

import os

class Txt2bunch(Base):

    """from batch txt to bunch
            variable：bunch = Bunch(label=[], content=[])
                                    label: true label, type: list[]
                                    content: directory all txt file, type: list[]
    """

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_data(self):
        self.bunch = Bunch(label=[], content=[])

        if os.path.exists(self.dic_config['raw_data']):
            files_list = os.listdir(self.dic_config['raw_data'])

            for file in files_list:
                self.bunch.label.append(file.split('_')[0])
                with open(self.dic_config['raw_data'] + '/' + file, 'r') as f:
                    container = []

                    for line in f.readlines():
                        a = map(int, list(line.replace('\n', '')))
                        container.extend(a)
                    self.bunch.content.append(container)

            self.bunch.content = np.array(self.bunch.content)

            one_hot = preprocessing.OneHotEncoder(sparse=False)
            self.bunch.label = np.array(self.bunch.label).reshape(-1, 1)
            self.bunch.label = self.bunch.label.tolist()
            self.bunch.label = one_hot.fit_transform(self.bunch.label)
            # self.bunch.label = self.bunch.label
            #
            # logging.info(self.bunch.label.shape)
            # logging.info(self.bunch.content.shape)


        self.logging.info('————txt to bunch successful————')

    def create(self):
        with open(self.dic_config['bunch_path'], 'wb') as f:
            pickle.dump(self.bunch, f)
        self.logging.info('file path: {}'.format(self.dic_config['bunch_path']))
