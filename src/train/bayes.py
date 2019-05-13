# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 上午10:57
# @Author  : Benqi

from base import Base
from sklearn.naive_bayes import MultinomialNB

import util.load_data as load_data

class Bayes(Base):

    """MultinomialNB: input(X, y):
                                X: np.array(sample_nums, vector_dim)
                                y: 1d array
                           example:
                                X = np.random.randint(5, size=(6, 100))
                                y = np.array([1, 2, 3, 4, 5, 6])

    """

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

        self.alpha = 0.001
        if 'alpha' in dic_config:
            self.alpha = dic_config['alpha']

    def load_data(self):
        self.bunch = load_data.read_bunch(self.data_path)
        self.logging.info(self.bunch.label)

    def train(self):
        self.model = MultinomialNB(alpha=self.alpha).fit(self.bunch.metrix, self.bunch.label)

