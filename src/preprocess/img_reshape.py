# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 下午2:29
# @Author  : Benqi

import os
import pickle

import cv2
from base import Base
import numpy as np

class Img_reshape(Base):
    """this class reshape image for train
    """

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)
        self.raw_data = dic_config['raw_data']
        self.pre_data = dic_config['pre_data']
        self.pickle_path = dic_config['pickle_path']

    def load_data(self, encode='utf-8'):
        def get_padding_size(image):
            h, w, _ = image.shape
            longest_edge = max(h, w)
            top, bottom, left, right = (0, 0, 0, 0)
            if h < longest_edge:
                dh = longest_edge - h
                top = dh // 2
                bottom = dh - top
            elif w < longest_edge:
                dw = longest_edge - w
                left = dw // 2
                right = dw - left
            else:
                pass
            return top, bottom, left, right

        def resize_image(data_path, pre_path, image_h, image_w):
            data_x = []
            data_y = []

            if not os.path.exists(pre_path):
                os.makedirs(pre_path)

            self.logging.info(data_path)
            for dir in os.listdir(data_path):
                self.logging.info(dir)
                dir_name = data_path + '/' + dir
                pre_name = os.path.join(pre_path, dir)

                if not os.path.exists(pre_name):
                    os.makedirs(pre_name)
                self.logging.info(dir_name)

                for i in os.listdir(dir_name):
                    fullname = dir_name + '/' + str(i)
                    # print(fullname)
                    img = cv2.imread(fullname)
                    top, bottom, left, right = get_padding_size(img)
                    img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    img = cv2.resize(img_pad, (image_h, image_w))

                    cv2.imwrite(pre_name + '/' + str(i), img)

                    data_x.append(img)
                    data_y.append(int(dir))
                    x = np.asarray(data_x)
                    y = np.asarray(data_y)

                    with open(self.pickle_path, 'wb') as f:
                        pickle.dump(x, f)
                        pickle.dump(y, f)

            return data_x, data_y

        self.data_x, self.data_y = resize_image(self.raw_data, self.pre_data, 28, 28)

    def create(self):
        self.logging.info('————reshape img successful————')
        self.logging.info('file path: %s' % (self.pre_data))