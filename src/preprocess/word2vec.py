# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 上午10:44
# @Author  : Benqi

from base import Base
import collections
import tensorflow as tf
import random
import numpy as np
import math


class Word2Vec(Base):
    """
    @:param pre_data 是一个txt文件，每个word占一行，其中每个word都是中文词汇
    @:param vector_path 是向量空间
    """

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_data(self):

        def read_txt(path):
            with open(path, 'r') as f:
                file = f.read()
            list = [i.strip() for i in file]
            return list

        self.word_list = read_txt(self.dic_config['pre_data'])

    def create(self):
        vocabulary_size = 50000

        def build_dataset(words):
            count = [['UNK', -1]]
            count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
            dictionary = dict()
            for word, _ in count:
                dictionary[word] = len(dictionary)
            data = list()
            unk_count = 0
            for word in words:
                if word in dictionary:
                    index = dictionary[word]
                else:
                    index = 0  # dictionary['UNK']
                    unk_count = unk_count + 1
                data.append(index)
            count[0][1] = unk_count
            reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
            return data, count, dictionary, reverse_dictionary

        data, count, dictionary, reverse_dictionary = build_dataset(self.word_list)
        # self.logging.info('Most common words (+UNK)', count[:5])
        # self.logging.info('Sample data', data[:10])

        data_index = 0

        def generate_batch(batch_size, num_skips, skip_window):
            data_index = 0
            assert batch_size % num_skips == 0
            assert num_skips <= 2 * skip_window
            batch = np.ndarray(shape=(batch_size), dtype=np.int32)
            labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
            span = 2 * skip_window + 1  # [ skip_window target skip_window ]
            buffer = collections.deque(maxlen=span)
            for _ in range(span):
                buffer.append(data[data_index])
                data_index = (data_index + 1) % len(data)
            for i in range(batch_size // num_skips):
                target = skip_window  # target label at the center of the buffer
                targets_to_avoid = [skip_window]
                for j in range(num_skips):
                    while target in targets_to_avoid:
                        target = random.randint(0, span - 1)
                    targets_to_avoid.append(target)
                    batch[i * num_skips + j] = buffer[skip_window]
                    labels[i * num_skips + j, 0] = buffer[target]
                buffer.append(data[data_index])
                data_index = (data_index + 1) % len(data)
            return batch, labels

        # self.logging.info('data:', [reverse_dictionary[di] for di in data[:8]])

        # for num_skips, skip_window in [(2, 1), (4, 2)]:
        #     batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
        #     self.logging.info('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
        #     self.logging.info('    batch:', [reverse_dictionary[bi] for bi in batch])
        #     self.logging.info('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

        num_steps = 6000

        batch_size = 128
        embedding_size = 128  # Dimension of the embedding vector.
        skip_window = 1  # How many words to consider left and right.
        num_skips = 2  # How many times to reuse an input to generate a label.

        num_sampled = 64  # Number of negative examples to sample.

        graph = tf.Graph()

        with graph.as_default(), tf.device('/cpu:0'):
            # Input data.
            train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            # Variables.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            softmax_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Model.
            # Look up embeddings for inputs.
            embed = tf.nn.embedding_lookup(embeddings, train_dataset)

            # Compute the softmax loss, using a sample of the negative labels each time.
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                           labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()

            self.logging.info('Initialized')
            average_loss = 0
            for step in range(num_steps):
                batch_data, batch_labels = generate_batch(
                    batch_size, num_skips, skip_window)
                feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += l
                if step % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    self.logging.info('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0

            embeddings_np = embeddings.eval()
            np.save(self.dic_config['vector_path'], embeddings_np)

            self.logging.info(embeddings_np[1, :])