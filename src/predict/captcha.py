# -*- coding: utf-8 -*-
# @Time    : 2018/11/21 下午12:25
# @Author  : Benqi
import os
import pickle

import cv2
import numpy as np
import tensorflow as tf

from base import Base

class Captcha(Base):
    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)
        self.vocab = dic_config['vocab']
        self.captcha_length = dic_config['captcha_length']
        self.predict_path = dic_config['predict_path']
        self.input_shape = dic_config['input_shape']
        self.vocab_length = len(self.vocab)

    def load_data(self):
        """
        load data from pickle or sample jpg
        return: feature:x, label: y
        """

        def standardize(x):

            """standardize function
            """
            return (x - x.mean()) / x.std()

        # create dataset
        with open(self.data_path, 'rb') as f:
            data_x = pickle.load(f)
            self.data_y = pickle.load(f)

        data_x = data_x.reshape(self.input_shape)
        self.data_x = standardize(data_x)
        self.logging.info(self.data_x)

    def load_model(self):
        self.logging.info(tf.__version__)
        self.sess = tf.Session()

        # load model
        self.saver = tf.train.import_meta_graph(self.model_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.abspath(os.path.join(self.dic_config['model_path'], '..'))))

        # get compute graph
        graph = tf.get_default_graph()

        """this for op sentence is very useful for debug when it raise not exist xxx on the graph

        for op in graph.get_operations():
            print(op.name)
        """
        # get operation name
        self.x = graph.get_tensor_by_name("holder/X:0")
        self.y = graph.get_tensor_by_name("output/Y_pred:0")

        self.logging.info('————model restore successful————')

    def predict(self):
        # run and get a result like y_pred
        self.result = self.sess.run(self.y, feed_dict={self.x: self.data_x})

        # result reshape to output
        self.result = np.reshape(self.result, (-1, self.captcha_length, self.vocab_length))
        self.logging.info(self.result.shape)
        result = np.argmax(self.result, axis=2)
        result = np.squeeze(result)

        result_list = []
        for word in result:
            char_list = []
            for i in word:
                char_list.append(self.vocab[i])

            output = "".join(char_list)
            result_list.append(output)
        self.logging.info(result_list)

        self.data_y = np.reshape(self.data_y, (-1, self.captcha_length, self.vocab_length))
        y_true = np.argmax(self.data_y, axis=2)
        y_true = np.squeeze(y_true)

        y_true_list = []
        for word in y_true:
            char_list = []
            for i in word:
                char_list.append(self.vocab[i])

            output = "".join(char_list)
            y_true_list.append(output)
        self.logging.info(y_true_list)

        n = 0
        for i in range(len(result_list)):
            if result_list[i] == y_true_list[i]:
                n += 1
        p = np.true_divide(n, len(result_list))
        self.logging.info(p)

    def dump(self):
        # np.savetxt(self.dic_config['predict_path'], self.predict_result, fmt='%s')
        pass