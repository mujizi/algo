# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 下午4:41
# @Author  : Benqi

import logging

# 准确率，召回率，F1
class Evalution():
    def __init__(self, dic_config={}):
        self.dic_config = dic_config

    def set_evaluate(self, model_name, eva_name):

        if eva_name == 'f1':
            import f1
            self.eva = f1.F1(self.dic_config[model_name][eva_name])

        elif eva_name == 'confuse_matrix':
            import confuse_matrix
            self.eva = confuse_matrix.Confuse_matrix(self.dic_config[model_name][eva_name])

        elif eva_name == 'roc':
            import roc
            self.eva = roc.Roc(self.dic_config[model_name][eva_name])

        elif eva_name == 'kappa':
            import kappa
            self.eva = kappa.Kappa(self.dic_config[model_name][eva_name])

    def run(self):
        for model_name in self.dic_config['model']:
            for eva_name in self.dic_config[model_name]['evaluate']:
                self.set_evaluate(model_name, eva_name)
                self.eva.load_y()
                self.eva.load_y_hat()
                self.eva.evaluate()
        logging.info("————评估结束————")