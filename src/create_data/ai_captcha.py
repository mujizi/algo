# -*- coding: utf-8 -*-
# @Time    : 2018/11/29 下午6:51
# @Author  : Benqi
import os
import pickle

from PIL import Image
from captcha.image import ImageCaptcha

import random

from base import Base
import numpy as np

class AICaptcha(Base):

    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)
        self.nums = dic_config['nums']
        self.data_path = dic_config['data_path']
        self.vocab = dic_config['vocab']
        self.captcha_length = dic_config['captcha_length']
        self.vocab_length = len(self.vocab)
        self.pickle_path = dic_config['pickle_path']

        self.logging.info(self.vocab)

    def generate_captcha(self, captcha_text):
        """
        get captcha text and np array
        :param captcha_text: source text
        :return: captcha image and array
        """
        image = ImageCaptcha()
        captcha = image.generate(captcha_text)
        captcha_image = Image.open(captcha)

        return captcha_image

    def text2vec(self, text):
        """
        text to one-hot vector
        :param text: source text
        :return: np array
        """
        if len(text) > self.captcha_length:
            return False
        vector = np.zeros(self.captcha_length * self.vocab_length)
        for i, c in enumerate(text):
            index = i * self.vocab_length + self.vocab.index(c)
            vector[index] = 1
        return vector

    def vec2text(self, vector):
        """
        vector to captcha text
        :param vector: np array
        :return: text
        """
        if not isinstance(vector, np.ndarray):
            vector = np.asarray(vector)
        vector = np.reshape(vector, [self.captcha_length, -1])
        text = ''
        for item in vector:
            text += self.vocab[np.argmax(item)]
        return text

    def get_random_text(self):
        text = ''
        for i in range(self.captcha_length):
            text += random.choice(self.vocab)
        return text

    def generate_data(self):
        self.logging.info('Generating Data...')
        data_x, data_y = [], []

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        for i in range(int(self.nums)):
            text = self.get_random_text()

            captcha = self.generate_captcha(text)

            vector = self.text2vec(text)
            captcha.save(self.data_path + '/' + str(text) + '.jpg')

            captcha = np.asarray(captcha, np.float32)
            data_x.append(captcha)
            data_y.append(vector)

        # self.logging.info(type(data_x[1]))

        x = np.asarray(data_x, np.float32)
        y = np.asarray(data_y, np.float32)

        with open(self.pickle_path, 'wb') as f:
            pickle.dump(x, f)
            pickle.dump(y, f)

        self.logging.info(x.shape)
        self.logging.info(y.shape)
