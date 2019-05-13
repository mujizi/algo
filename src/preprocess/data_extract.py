# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 下午2:16
# @Author  : Benqi
import pandas as pd
from base import Base
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class Data_extract(Base):
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)
        self.raw_data = dic_config['raw_data']
        self.csv_path = dic_config['csv_path']
        self.task = dic_config['task']
        self.col_name = dic_config['col_name']
        self.row_name = dic_config['row_name']
        self.values = dic_config['values']
        self.col_nums = dic_config['col_nums']

    def extract_col(self, dataset, col_name):
        return dataset[col_name]

    def extract_row(self, dataset, row_name):
        return dataset.loc[row_name]

    def drop_na(self, dataset):
        return dataset.dropna()

    def fill_na(self, dataset, col_name, values):
        return dataset.fillna({col_name:values})

    def normalization(self, dataset):
        column_names = [i for i in dataset]
        values = dataset.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        df = pd.DataFrame(scaled, columns=column_names)
        return df

    def label_encoder(self, dataset, col_nums):
        col_names = [columns for columns in dataset]
        encoder = LabelEncoder()
        values = dataset.values
        values[:, col_nums] = encoder.fit_transform(values[:, col_nums])
        df = pd.DataFrame(values, columns=col_names)
        return df

    def load_data(self):
        self.dataset = pd.read_csv(self.raw_data, index_col=0)

    def create(self):
        if self.task == 'extract_col':
            self.result = self.extract_col(self.dataset, self.col_name)

        if self.task == 'extract_row':
            self.result = self.extract_row(self.dataset, self.row_name)

        if self.task == 'drop_na':
            self.result = self.drop_na(self.dataset)

        if self.task == 'fill_na':
            self.result = self.fill_na(self.dataset, self.col_name, self.values)

        if self.task == 'normalization':
            self.result = self.normalization(self.dataset)

        if self.task == 'label_encoder':
            self.result = self.label_encoder(self.dataset, self.col_nums)

        self.result.to_csv(self.csv_path)