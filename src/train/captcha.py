# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 下午6:33
# @Author  : Benqi
import os
import pickle
import random

import tensorflow as tf
import numpy as np

from base import Base
from os.path import join

class Captcha(Base):

    """this class is Verification code recognition

       it is a Convolution neural network

       used tensorflow layer API
    """
    sess = None
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)
        self.input_shape = dic_config['input_shape']
        self.sample_nums = dic_config['sample_nums']

        self.batch_size = dic_config['batch_size']

        self.epoch_nums = dic_config['epoch_nums']

        self.keep_prob = dic_config['keep_prob']
        self.learning_rate = dic_config['learning_rate']

        self.vocab = dic_config['vocab']
        self.captcha_length = dic_config['captcha_length']
        self.vocab_length = len(self.vocab)

    def load_data(self):

        """
        load data from pickle
        :return:
        """

        def standardize(x):

            """standardize function
            """
            return (x - x.mean()) / x.std()

        with open(join(self.data_path), 'rb') as f:
            data_x = pickle.load(f)
            self.data_y = pickle.load(f)
            self.data_y - np.float32(self.data_y)

            data_x = data_x.reshape((-1, 160, 60, 3))
            self.data_x = standardize(data_x)
            self.data_x = np.float32(self.data_x)

        self.logging.info('data X Shape:   %s   data Y Shape:   %s' % (self.data_x.shape, self.data_y.shape))

    def train(self):
        # cnn_network
        def network(data_x, data_y, keep_p, learning_rate, batch_size, sample_nums, epoch_nums):

            # create batch data
            def create_batch_dataset(dataset_x, dataset_y, batch_size, sample_nums):
                batch_x = np.zeros([batch_size, 160, 60, 3])
                batch_y = np.zeros([batch_size, 40])

                for i in range(batch_size):
                    batch_x[i, :, :, :] = dataset_x[random.randint(0, sample_nums - 1), :, :, :]
                    batch_y[i, :] = dataset_y[random.randint(0, sample_nums - 1), :]

                return batch_x, batch_y

            batch_x, batch_y = create_batch_dataset(data_x, data_y, batch_size, sample_nums)
            self.logging.info(batch_x.shape)

            batch_y = np.reshape(batch_y, (-1, self.vocab_length * self.captcha_length))
            self.logging.info(batch_y.shape)

            # holder
            with tf.name_scope("holder"):
                X = tf.placeholder(tf.float32, shape=(None, 160, 60, 3), name='X')

            with tf.name_scope("convlution"):
                self.conv1 = tf.layers.conv2d(X, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
                self.pool1 = tf.layers.max_pooling2d(self.conv1, pool_size=2, strides=2, padding='same')

                self.conv2 = tf.layers.conv2d(self.pool1, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
                self.pool2 = tf.layers.max_pooling2d(self.conv2, pool_size=2, strides=2, padding='same')

                self.conv3 = tf.layers.conv2d(self.pool2, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
                self.pool3 = tf.layers.max_pooling2d(self.conv3, pool_size=2, strides=2, padding='same')

            with tf.name_scope("flatten"):
                self.y = tf.layers.flatten(self.pool3)

            with tf.name_scope("dense"):
                self.y = tf.layers.dense(self.y, 1024, activation=tf.nn.relu)
                self.y = tf.layers.dropout(self.y, rate=keep_p)
                self.y = tf.layers.dense(self.y, self.vocab_length * self.captcha_length)

            with tf.name_scope("output"):
                y_pred = tf.reshape(self.y, [-1, self.vocab_length * self.captcha_length], name='Y_pred')
                y_pred = np.float32(y_pred)

            with tf.name_scope("cross_entropy"):
                cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_y, logits=y_pred))

                y_pred = tf.reshape(y_pred, (-1, 10, 4))
                batch_y = tf.reshape(batch_y, (-1, 10, 4))
                max_index_predict = tf.argmax(y_pred, axis=1)
                max_index_label = tf.argmax(batch_y, axis=1)

                correct_predict = tf.equal(max_index_predict, max_index_label)

            accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name='Accuracy')

            train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

            # saver
            saver = tf.train.Saver()

            # global step
            gstep = 0

            # init variables
            self.sess = tf.Session()

            # writer graph
            writer = tf.summary.FileWriter(os.path.abspath(os.path.join(self.model_path, '../log')), self.sess.graph)
            writer.close()

            self.sess.run(tf.global_variables_initializer())

            # run cross and accuracy
            for step in range(epoch_nums):
                loss, acc, _ = self.sess.run([cross_entropy, accuracy, train_op], feed_dict={X: batch_x})

                if step % 10 == 0:
                    self.logging.info('Train Loss: {}'.format(loss))
                    self.logging.info('Accuracy: {}'.format(acc))

        network(self.data_x, self.data_y, self.keep_prob, self.learning_rate, self.batch_size, self.sample_nums, self.epoch_nums)

    def dump(self):
        if self.sess:
            m_saver = tf.train.Saver()
            m_saver.save(self.sess, self.model_path)
