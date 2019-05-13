# -*- coding: utf-8 -*-
# @Time    : 2018/11/21 下午12:25
# @Author  : Benqi
import cv2
import pickle

import keras
import numpy as np

from sklearn.model_selection import train_test_split
from base import Base

class Ai_face(Base):
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)
        self.format = dic_config['format']
        self.input_shape = dic_config['input_shape']
        self.data_path = dic_config['data_path']
        self.model_path = dic_config['model_path']

    def load_data(self):

        def standardize(x):
            """standardize function
            """
            return (x - x.mean()) / x.std()

        if self.format == 'sample':
            img = cv2.imread(self.data_path)
            img = img.reshape(self.input_shape)
            self.data = img

        else:
            with open(self.data_path, 'rb') as f:
                data_x = pickle.load(f)
                data_y = pickle.load(f)

            data_x = data_x.reshape(self.input_shape)
            data_x = standardize(data_x)

            _, self.data, _, _= train_test_split(data_x, data_y, test_size=0.4, random_state=40)

    def load_model(self):
        self.model = keras.models.load_model(self.model_path)

    def predict(self):
        self.predict_result = self.model.predict(self.data)
        self.predict_result = np.squeeze(self.predict_result)
        self.predict_result = np.argmax(self.predict_result) + 1
        self.logging.info(self.predict_result)

    def dump(self):
        # np.savetxt(self.dic_config['predict_path'], str(self.predict_result), fmt='%s')
        pass