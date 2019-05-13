# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 下午6:33
# @Author  : Benqi
import os
import math
import pickle
import tensorflow as tf

from base import Base
from os.path import join
from sklearn.model_selection import train_test_split

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

        self.train_batch_size = dic_config['train_batch_size']
        self.dev_batch_size = dic_config['dev_batch_size']
        self.test_batch_size = dic_config['test_batch_size']

        self.epoch_nums = dic_config['epoch_nums']
        self.epochs_per_dev = dic_config['epochs_per_dev']
        self.epochs_per_save = dic_config['epochs_per_save']
        self.steps_per_print = dic_config['steps_per_print']

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
            data_y = pickle.load(f)

            data_x = data_x.reshape((-1, 160, 60, 3))
            data_x = standardize(data_x)

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(data_x, data_y, test_size=0.4, random_state=40)
        self.dev_x, self.test_x, self.dev_y, self.test_y, = train_test_split(self.test_x, self.test_y, test_size=0.5, random_state=40)

        self.logging.info('Train X Shape:   %s   Train Y Shape:   %s' % (self.train_x.shape, self.train_y.shape))
        self.logging.info('Test X Shape:   %s   Test Y Shape:   %s' % (self.test_x.shape, self.test_y.shape))
        self.logging.info('Dev X Shape:   %s   Dev Y Shape:   %s' % (self.dev_x.shape, self.dev_y.shape))

    def train(self):
        train_steps = math.ceil(self.train_x.shape[0] / self.train_batch_size)
        dev_steps = math.ceil(self.dev_x.shape[0] / self.dev_batch_size)

        global_step = tf.Variable(-1, trainable=False, name='global_step')

        train_dataset = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y)).shuffle(self.sample_nums)
        train_dataset = train_dataset.batch(self.train_batch_size)

        dev_dataset = tf.data.Dataset.from_tensor_slices((self.dev_x, self.dev_y))
        dev_dataset = dev_dataset.batch(self.dev_batch_size)

        # feed holder
        with tf.name_scope("holder"):
            handle = tf.placeholder(tf.string, shape=[], name='Handle')
            keep_prob = tf.placeholder(tf.float32, [], name='Keep_prob')

            iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

            self.x, y_label = iterator.get_next(name='Iterator_Out')
            y_true = tf.reshape(y_label, [-1, self.vocab_length])

            train_tupple = train_dataset.make_initializable_iterator()
            dev_tupple = dev_dataset.make_initializable_iterator()

        with tf.name_scope("convlution"):
            self.conv1 = tf.layers.conv2d(self.x, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='Conv_input')
            self.pool1 = tf.layers.max_pooling2d(self.conv1, pool_size=2, strides=2, padding='same')

            self.conv2 = tf.layers.conv2d(self.pool1, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
            self.pool2 = tf.layers.max_pooling2d(self.conv2, pool_size=2, strides=2, padding='same')

            self.conv3 = tf.layers.conv2d(self.pool2, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
            self.pool3 = tf.layers.max_pooling2d(self.conv3, pool_size=2, strides=2, padding='same')

        with tf.name_scope("flatten"):
            self.y = tf.layers.flatten(self.pool3)

        with tf.name_scope("dense"):
            self.y = tf.layers.dense(self.y, 1024, activation=tf.nn.relu)
            self.y = tf.layers.dropout(self.y, rate=keep_prob)
            self.y = tf.layers.dense(self.y, self.vocab_length * self.captcha_length)

        with tf.name_scope("output"):
            y_pred = tf.reshape(self.y, [-1, 10], name='Y_pred')


        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true))

            max_index_predict = tf.argmax(y_pred, axis=-1)
            max_index_label = tf.argmax(y_true, axis=-1)

            correct_predict = tf.equal(max_index_predict, max_index_label)

        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name='Accuracy')

        train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cross_entropy, global_step=global_step)

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
        train_handle = self.sess.run(train_tupple.string_handle())
        dev_handle = self.sess.run(dev_tupple.string_handle())

        for epoch in range(self.epoch_nums):
            tf.train.global_step(self.sess, global_step_tensor=global_step)

            # train


            self.sess.run(train_tupple.initializer)
            for step in range(int(train_steps)):
                loss, acc, gstep, _ = self.sess.run([cross_entropy, accuracy, global_step, train_op],
                                               feed_dict={handle: train_handle, keep_prob: self.keep_prob})

                if step % self.steps_per_print == 0:
                    self.logging.info('Train Loss: {}'.format(loss))
                    self.logging.info('Accuracy: {}'.format(acc))

            if epoch % self.epochs_per_dev == 0:
                # dev
                self.sess.run(dev_tupple.initializer)
                for step in range(int(dev_steps)):
                    if step % self.steps_per_print == 0:
                        self.logging.info('Dev Accuracy: {}'.format(self.sess.run(accuracy, feed_dict={handle: dev_handle, keep_prob: 1})))

    def dump(self):
        if self.sess:
            m_saver = tf.train.Saver()
            m_saver.save(self.sess, self.model_path)
