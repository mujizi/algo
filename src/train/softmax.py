# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 下午2:48
# @Author  : Benqi
import numpy as np

from base import Base
import util.tools as tools
import tensorflow as tf

class Softmax(Base):

    sess = None

    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)
        self.num_iter = dic_config['num_iter']
        self.learning_rate = dic_config['learning_rate']

    def load_data(self):
        bunch = tools.read_bunch(self.data_path)

        self.X = np.array(bunch.content)
        self.Y = np.array(bunch.label)

        self.logging.info(self.Y.shape)
        self.logging.info(self.X.shape)

    def train(self):
        dim = self.X.shape[1]    # dim
        m = self.Y.shape[1]      # target_nums

        with tf.name_scope('Input'):
            x = tf.placeholder(tf.float32, [None, dim], name='X_place_holder')
            y = tf.placeholder(tf.float32, [None, m], name='Y_place_holder')

        with tf.name_scope('Inference'):
            w = tf.Variable(tf.random_normal([dim, m], stddev=0.01), name='Weights')
            b = tf.Variable(tf.zeros([m]), name='Bias')
            logits = tf.matmul(x, w) + b
            y_pred = tf.nn.softmax(logits=logits, name='Y_pred')

        with tf.name_scope('Loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name='Cross_entropy')
            loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('Optimization'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            # train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)
            # init = tf.global_variables_initializer()

        with tf.name_scope('Evaluate'):
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for i in range(0, self.num_iter):
            _, loss_batch = self.sess.run([optimizer, loss], feed_dict={x: self.X, y: self.Y})

            if self.num_iter % 100 == 0:
                self.logging.info('step:train_loss:{}'.format(loss_batch))

    def dump(self):
        if self.sess:
            m_saver = tf.train.Saver()
            m_saver.save(self.sess, self.model_path)
            tf.Session.close()
