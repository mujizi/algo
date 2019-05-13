# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 下午3:13
# @Author  : Benqi
from base import Base

import pandas as pd

class Series2supervised(Base):
    """Change series dataset to supervised dataset, is used to train rnns model
        @:param n_in: (t-n ... t-1) time stamp
        @:param n_out: (t ... t+n) time stamp
    """

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)
        self.raw_data = dic_config['raw_data']
        self.csv_path = dic_config['csv_path']
        self.n_in = dic_config['n_in']
        self.n_out = dic_config['n_out']
        self.dropna = dic_config['dropna']

    def series2supervised(self, data, n_in=1, n_out=1, dropna=True):
        n_features = 1 if type(data) is list else data.shape[1]

        df = pd.DataFrame(data)
        cols, names = [], []
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i))for j in range(n_features)]
        # forcast sequence (t, t+1, t+2 ...)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_features)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_features)]

        result = pd.concat(cols, axis=1)
        result.columns = names
        if dropna:
            result.dropna(inplace=True)

        return result

    def load_data(self):
        self.dataset = pd.read_csv(self.raw_data, index_col=0)

    def create(self):
        self.result = self.series2supervised(self.dataset, self.n_in, self.n_out, self.dropna)
        self.result.to_csv(self.csv_path)



