# -*- coding: utf-8 -*-
# @Time    : 2018/11/3 下午2:51
# @Author  : Benqi

import pandas as pd

from base import Base
import util.tools as tools

class Ai_xgboost(Base):

    """data is pandas type. like data list

    """

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_model(self):
        load = tools.Access_model()
        self.model = load.load_model(self.dic_config['model_path'])

    def load_data(self):
        self.x = pd.read_csv(self.dic_config['data_path'])

    def predict(self):
        y = self.model.predict(self.x)
        self.logging.info('————预测结束————')
        self.predict_result = y
        return y
