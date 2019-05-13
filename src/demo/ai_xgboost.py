# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 下午7:38
# @Author  : Benqi
import sys
sys.path.append('../')

import pandas as pd
import util.tools as tools

class Ai_xgboost():
    """data is pandas type. like data list

    """
    def load_model(self, model_path):
        load = tools.Access_model()
        self.model = load.load_model(model_path)

    def load_data(self, data_path):
        self.x = pd.read_csv(data_path)
        return self.x

    def predict(self, data_path):
        self.x = self.load_data(data_path)
        y = self.model.predict(self.x)
        print('————预测结束————')
        self.predict_result = y
        return y

if __name__ == '__main__':
    ai_xgboost = Ai_xgboost()
    ai_xgboost.load_model('')
    ai_xgboost.predict('/Users/ouhon/PycharmProjects/ai_algo/data/ai_xgboost/raw/test.csv')