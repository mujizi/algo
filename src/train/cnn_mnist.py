# -*- coding: utf-8 -*-
# @Time    : 2018/11/16 下午4:37
# @Author  : Benqi

import tensorflow as tf
from tensorflow import keras

from base import Base

class Cnn_mnist(Base):
    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)
        self.learning_rate = dic_config['learning_rate']
        self.keep_prob = dic_config['keep_prob']

    def load_data(self):
        fashion_mnist = keras.datasets.fashion_mnist

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()

        self.logging.info(self.train_images.shape)
        self.logging.info(len(self.train_labels))

    def train(self):
        def weight(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, w):
            return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        def reshape_image(x):
            x_image = tf.reshape(x, [-1, 28, 28, 3])
            return x_image

        with tf.name_scope('Input'):
            X = tf.placeholder(tf.float32, name='X_place_holder')
            Y = tf.placeholder(tf.float32, name= 'Y_place_holder')

        with tf.name_scope('Network'):
            w_conv1 = weight([3, 3, 3, 32])
            b_conv1 = bias([32])

            conv1 = tf.nn.relu(conv2d(X, w_conv1) + b_conv1)
            pool1 = max_pool(conv1)

            w_conv2 = weight([3, 3, 32, 64])
            b_conv2 = bias([64])

            conv2 = tf.nn.relu(conv2d(pool1, w_conv2) + b_conv2)
            pool2 = max_pool(conv2)

            w_fc1 = weight([7*7*64, 1024])
            b_fc1 = bias([1024])

            pool_flat = tf.reshape(pool2, [-1, 7*7*64])
            fc1 = tf.nn.relu(tf.matmul(pool_flat, w_fc1) + b_fc1)

            keep_preb = tf.placeholder(tf.float32)
            fc1_dropout = tf.nn.dropout(fc1, keep_preb)

            w_fc2 = weight([1024, 10])
            b_fc2 = bias([10])

            y_out = tf.matmul(fc1_dropout, w_fc2) + b_fc2
            y_pred = tf.nn.softmax(logits=y_out, name='Y_pred')

        with tf.name_scope('Loss'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y,logits= y_pred,
                                                                                   name='Cross_entropy'))

        with tf.name_scope('Optimization'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        with tf.name_scope('Evaluate'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1),tf.argmax()), tf.float32))

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for i in range(3000):
                if i % 100 == 0:
                    sess.run(optimizer, feed_dict={X: self.train_images, Y: self.train_labels})
                    train_accuracy = accuracy.eval(feed_dict={X: self.train_images,
                                                              Y: self.train_labels,
                                                              keep_preb: self.keep_prob})
                    self.logging.info(train_accuracy)

            saver = tf.train.Saver()
            saver.save(sess, self.model_path)

    def dump(self):
        self.logging.info('model path: %s' % (self.model_path))