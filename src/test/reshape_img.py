# -*- coding: utf-8 -*-
# @Time    : 2019/2/21 下午1:27
# @Author  : Benqi

import os
import pickle

import cv2
import numpy as np

class Img_reshape():
    """this class reshape image for train
    """
    def __init__(self, raw_path, pre_data):
        self.raw_data = raw_path
        self.pre_data = pre_data

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

            for dir in os.listdir(data_path):
                dir_name = data_path + '/' + dir
                pre_name = os.path.join(pre_path, dir)

                if not os.path.exists(pre_name):
                    os.makedirs(pre_name)

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

            return data_x, data_y

        self.data_x, self.data_y = resize_image(self.raw_path, self.pre_data, 28, 28)


if __name__ == '__main__':
    raw_path = '9.jpg'
    pre_data = 're9.jpg'
    reshape = Img_reshape(raw_path, pre_data)

