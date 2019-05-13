# -*- coding: utf-8 -*-
# @Time    : 2018/11/21 下午12:25
# @Author  : Benqi
import os
import cv2
import numpy as np
import tensorflow as tf

class Captcha():

    vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    captcha_length = 4
    vocab_length = len(vocab)

    def load_data(self, file_path):
        """
        load data from pickle or sample jpg
        return: feature:x, label: y
        """

        def standardize(x):
            """standardize function
            """
            return (x - x.mean()) / x.std()

        # create dataset
        img = cv2.imread(file_path)
        img = img.reshape((-1, 160, 60, 3))
        img = standardize(img)

        return np.float32(img)

    def load_model(self, model_path):
        print(tf.__version__)
        self.sess = tf.Session()

        # load model
        self.saver = tf.train.import_meta_graph(model_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.abspath(os.path.join(model_path, '..'))))

        # get compute graph
        graph = tf.get_default_graph()

        """this for op sentence is very useful for debug when it raise not exist xxx on the graph

        for op in graph.get_operations():
            print(op.name)
        """
        # get operation name
        self.x = graph.get_tensor_by_name("holder/X:0")
        self.y = graph.get_tensor_by_name("output/Y_pred:0")

        print('————model restore successful————')

    def predict(self, file_path):
        img = self.load_data(file_path)

        # run and get a result like y_pred
        self.result = self.sess.run(self.y, feed_dict={self.x: img})
        print(self.result)

        # result reshape to output
        self.result = np.reshape(self.result, (-1, self.captcha_length, self.vocab_length))
        print(self.result)

        result = np.argmax(self.result, axis=2)
        result = np.squeeze(result)

        result_list = []
        for i in result:
            result_list.append(self.vocab[i])

        output = "".join(result_list)
        return output

if __name__ == '__main__':
    captcha = Captcha()
    captcha.load_model('/Users/ouhon/PycharmProjects/ai_algo/data/captcha/train/666.meta')
    result = captcha.predict('/Users/ouhon/PycharmProjects/ai_algo/data/captcha/create_data/acea.jpg')
    print('predict result:   %s' % (result))