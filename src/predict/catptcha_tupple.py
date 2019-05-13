# -*- coding: utf-8 -*-
# @Time    : 2018/11/21 下午12:25
# @Author  : Benqi
import os
import cv2
import pickle
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from base import Base

class Captcha(Base):
    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)
        self.format = dic_config['format']
        self.vocab = dic_config['vocab']
        self.captcha_length = dic_config['captcha_length']
        self.predict_path = dic_config['predict_path']
        self.input_shape = dic_config['input_shape']
        self.keep_prob = dic_config['keep_prob']
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
        if self.format == 'sample':
            img = cv2.imread(self.data_path)
            y = np.random.random((1, self.captcha_length * self.vocab_length))
            img = img.reshape(self.input_shape)
            img = standardize(img)
            img = np.float32(img)
            y = np.float32(y)

            self.dataset = tf.data.Dataset.from_tensor_slices((img, y))
            self.dataset = self.dataset.batch(1)
            self.tupple = self.dataset.make_initializable_iterator()

        else:
            with open(os.path.join(self.data_path), 'rb') as f:
                data_x = pickle.load(f)
                data_y = pickle.load(f)

                data_x = data_x.reshape(self.input_shape)
                data_x = standardize(data_x)

            # slice feature and label
            _, self.test_x, _, self.test_y = train_test_split(data_x, data_y, test_size=0.4, random_state=40)

            self.logging.info(' X Shape:   %s    Y Shape:   %s' % (self.test_x.shape, self.test_y.shape))

            self.dataset = tf.data.Dataset.from_tensor_slices((self.test_x, self.test_y)).shuffle(1)
            self.dataset = self.dataset.batch(40)
            self.tupple = self.dataset.make_initializable_iterator()

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
        self.y = graph.get_tensor_by_name("output/Y_pred:0")
        self.x = graph.get_tensor_by_name("holder/Handle:0")
        self.prob = graph.get_tensor_by_name('holder/Keep_prob:0')

        self.logging.info('————model restore successful————')

    def predict_batch(self):
        pass

    def predict(self):
        # init tupple
        self.sess.run(self.tupple.initializer)

        # join in string_handle
        test_handle = self.sess.run(self.tupple.string_handle())

        # run and get a result like y_pred
        self.result = self.sess.run(self.y, feed_dict={self.x: test_handle, self.prob: self.keep_prob})

        # result reshape to output
        self.result = np.reshape(self.result, (-1, self.captcha_length, self.vocab_length))
        self.logging.info(self.result.shape)
        result = np.argmax(self.result, axis=2)
        result = np.squeeze(result)

        result_list = []
        for i in result:
            result_list.append(self.vocab[i])

        output = "".join(result_list)

        self.logging.info(output)

    def dump(self):
        # np.savetxt(self.dic_config['predict_path'], self.predict_result, fmt='%s')
        pass