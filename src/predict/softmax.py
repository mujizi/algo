# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 下午2:41
# @Author  : Benqi
import os

import tensorflow as tf
import numpy as np

from base import Base
import util.load_data as load_data

class Softmax(Base):
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_model(self):
        self.logging.info(tf.__version__)
        self.sess = tf.Session()

        self.logging.info(self.model_path)
        saver = tf.train.import_meta_graph(self.model_path)
        saver.restore(self.sess, tf.train.latest_checkpoint(os.path.abspath(os.path.join(self.dic_config['model_path'], '..'))))

        graph = tf.get_default_graph()

        """this for op sentence is very useful for debug when it raise not exist xxx on the graph
        
        for op in graph.get_operations():
            print(op.name)
        """

        self.x = graph.get_tensor_by_name("Input/X_place_holder:0")
        self.y = graph.get_tensor_by_name("Input/Y_place_holder:0")
        self.y_pred = graph.get_tensor_by_name('Inference/Y_pred:0')

        self.logging.info('————model restore successful————')

    def load_data(self):
        bunch = load_data.read_bunch(self.dic_config['data_path'])
        self.x_test = np.array(bunch.content)
        self.y_true = bunch.label
        self.y_true_list = []
        for i in range(0, self.y_true.shape[0]):
            self.y_true_list.append(np.argmax(self.y_true[i]))

        self.logging.info(len(self.y_true_list))

    def predict(self):
        self.pred =self.sess.run(self.y_pred, feed_dict={self.x:self.x_test})
        self.pred_list = []

        a = 0
        for i in range(0, self.pred.shape[0]):
            self.pred_list.append(np.argmax(self.pred[i]))

            if self.pred_list[i] == self.y_true_list[i]:
                a += 1

        self.logging.info('predict correct nums : %s' % (a))
        accuracy = np.true_divide(a, len(self.pred_list))
        self.logging.info('accuracy: %s' % (accuracy))

    def dump(self):
        np.savetxt(self.dic_config['predict_path'], self.pred_list, fmt='%s', delimiter=',')