# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 上午10:58
# @Author  : Benqi
import pickle
import logging
import pandas as pd
from scipy import sparse
from xgboost import XGBClassifier

from base import Base

class Ai_xgboost(Base):

    """ X : array_like
            Feature matrix
        y : array_like
            Labels
        we use pandas.DataFrame
    """

    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)
        self.sparse_path = dic_config['sparse_path']
        self.silent = dic_config['silent']
        self.nthread = dic_config['nthread']
        self.learning_rate = dic_config['learning_rate']
        self.min_child_weight = dic_config['min_child_weight']
        self.max_depth = dic_config['max_depth']
        self.gamma = dic_config['gamma']
        self.subsample = dic_config['subsample']
        self.max_delta_step = dic_config['max_delta_step']
        self.colsample_bytree = dic_config['colsample_bytree']
        self.reg_lambda = dic_config['reg_lambda']
        self.reg_alpha = dic_config['reg_alpha']
        self.scale_pos_weight = dic_config['scale_pos_weight']
        self.n_estimators = dic_config['n_estimators']
        self.seed = dic_config['seed']
        self.eval_metric = dic_config['eval_metric']

    def load_data(self):
        self.train_data = pd.read_csv(self.data_path)

        self.y = self.train_data['label'].values

        self.x = sparse.load_npz(self.sparse_path)

        logging.info(type(self.x))
        logging.info(len(self.y))

    def train(self):
        xgb = XGBClassifier(silent=self.silent,
                            nthread=self.nthread,
                            learning_rate=self.learning_rate,
                            min_child_weight=self.min_child_weight,
                            max_depth=self.max_depth,
                            gamma= self.gamma,
                            subsample=self.subsample,
                            max_delta_step=self.max_delta_step,
                            colsample_bytree=self.colsample_bytree,
                            reg_lambda=self.reg_lambda,
                            reg_alpha=self.reg_alpha,
                            scale_pos_weight=self.scale_pos_weight,
                            objective= 'reg:logistic',
                            n_estimators=self.n_estimators,
                            seed=self.seed,
                            eval_metric=self.eval_metric
        )

        self.model = xgb.fit(self.x, self.y, eval_metric='auc')

    def dump(self):
        with open(self.model_path, 'w') as f:
            pickle.dump(self.model, f)
