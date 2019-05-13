# -*- coding: utf-8 -*-
# @Time    : 2018/11/29 下午1:35
# @Author  : Benqi
import os
import numpy as np
from PIL import Image

from sklearn.utils import Bunch
from sklearn.preprocessing import OneHotEncoder

from base import Base
import util.tools as tools

class Img2bunch(Base):
    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)
        self.data_path = dic_config['data_path']
        self.bunch_path = dic_config['bunch_path']

    def load_data(self):
        full_file_name = []
        image_data = []
        y_true_list = []
        bunch = Bunch(x=[], y=[])

        for i in os.listdir(self.data_path):
            self.logging.info(type(i))

            for filename in os.listdir(self.data_path + '/' + str(i)):

                full_file_name.append(self.data_path + '/' + str(i) + '/' + filename)

                y_true_list.append(i)
                img = Image.open(self.data_path + '/' + str(i) + '/' + filename)

                img = np.array(img)

                image_data.append(img)

        y_true_list = np.array(y_true_list)
        y_true_list = y_true_list.reshape((-1, 1))

        one = OneHotEncoder()
        y_true_list = one.fit_transform(y_true_list)

        image_data = np.array(image_data).reshape((-1, 40, 32, 1))
        self.image_data = image_data
        self.y_true = y_true_list.toarray()

        self.logging.info(self.image_data.shape)
        self.logging.info(self.y_true.shape)

        self.bunch = bunch
        self.bunch.x = self.image_data
        self.bunch.y = self.y_true

    def create(self):
        tools.write_bunch(self.bunch_path, self.bunch)
