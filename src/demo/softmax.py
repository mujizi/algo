# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 下午7:32
# @Author  : Benqi
import sys
sys.path.append('../')

import os
import numpy as np
import tensorflow as tf

import util.load_data as load_data


class Softmax():
    def load_model(self, model_path):
        print(tf.__version__)
        self.sess = tf.Session()

        print(model_path)
        saver = tf.train.import_meta_graph(model_path)
        saver.restore(self.sess,
                      tf.train.latest_checkpoint(os.path.abspath(os.path.join(model_path, '..'))))

        graph = tf.get_default_graph()

        """this for op sentence is very useful for debug when it raise not exist xxx on the graph

        for op in graph.get_operations():
            print(op.name)
        """

        self.x = graph.get_tensor_by_name("Input/X_place_holder:0")
        self.y = graph.get_tensor_by_name("Input/Y_place_holder:0")
        self.y_pred = graph.get_tensor_by_name('Inference/Y_pred:0')

        print('————model restore successful————')

    def load_data(self, data_path):
        bunch = load_data.read_bunch(data_path)
        self.x_test = np.array(bunch.content)
        self.y_true = bunch.label
        self.y_true_list = []
        for i in range(0, self.y_true.shape[0]):
            self.y_true_list.append(np.argmax(self.y_true[i]))

        print(len(self.y_true_list))
        return self.x_test

    def predict(self, data_path):
        self.x_test = self.load_data(data_path)
        self.pred = self.sess.run(self.y_pred, feed_dict={self.x: self.x_test})
        self.pred_list = []

        a = 0
        for i in range(0, self.pred.shape[0]):
            self.pred_list.append(np.argmax(self.pred[i]))

            if self.pred_list[i] == self.y_true_list[i]:
                a += 1

        print('predict correct nums : %s' % (a))
        accuracy = np.true_divide(a, len(self.pred_list))
        print('accuracy: %s' % (accuracy))

if __name__ == '__main__':
    softmax = Softmax()
    softmax.load_model('/Users/ouhon/PycharmProjects/ai_algo/data/softmax/train/softmax.m.meta')
    softmax.predict('/Users/ouhon/PycharmProjects/ai_algo/data/softmax/preprocess/test_digits.dat')