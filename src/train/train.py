# -*- coding: utf-8 -*-
# @Time    : 2018/9/1 下午4:28
# @Author  : Benqi

import logging

class Trainer():
    def __init__(self, dic_config={}):
        self.dic_config = dic_config

    def set_model(self, model_name):
        if model_name == 'bayes':
            import bayes
            self.model = bayes.Bayes(self.dic_config[model_name])

        elif model_name == 'logistic':
            import logistic
            self.model = logistic.Logistic(self.dic_config[model_name])

        elif model_name == 'softmax':
            import softmax
            self.model = softmax.Softmax(self.dic_config[model_name])

        elif model_name == 'softmax_keras':
            import softmax_keras
            self.model = softmax_keras.Softmax_keras(self.dic_config[model_name])

        elif model_name == 'ai_xgboost':
            import ai_xgboost
            self.model = ai_xgboost.Ai_xgboost(self.dic_config[model_name])

        elif model_name == 'cnn_keras':
            import cnn_keras
            self.model = cnn_keras.Cnn_keras(self.dic_config[model_name])

        elif model_name == 'captcha':
            import captcha
            self.model = captcha.Captcha(self.dic_config[model_name])

        elif model_name == 'ai_face':
            import ai_face
            self.model = ai_face.Ai_face(self.dic_config[model_name])

        elif model_name == 'ai_lstm':
            import ai_lstm
            self.model = ai_lstm.Ai_lstm(self.dic_config[model_name])

    def load_data(self):
        self.model.load_data()

    def train(self):
        logging.info('model train start')
        self.model.train()
        logging.info('model train end')

    def dump(self):
        self.model.dump()

    def run(self):
        for model_name in self.dic_config['model']:
            self.set_model(model_name)
            self.load_data()
            self.train()
            self.dump()