# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 下午4:18
# @Author  : Benqi

CONFIG = {
    'name': 'captcha',
    'task': ['predict'],

    'create_data': {
        'task': ['captcha'],
        'captcha': {'data_path': '',
                    'pickle_path': 'data.pkl',
                    'vocab': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                    'captcha_length': 4,
                    'nums': '100'}
    },

    'train': {
        'model': ['captcha'],

        'captcha': {'data_path': 'data.pkl',
                    'model_path': 'model.h5',
                    'input_shape': (160, 60, 3)}
    },

    'predict': {
        'model': ['captcha'],
        'captcha': {'model_path': 'model.h5',
                    'input_shape': (-1, 160, 60, 3),
                    'format': 'sa',
                    'data_path': 'data.pkl',
                    'vocab': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                    'captcha_length': 4,
                    'predict_path': 'captcha.txt'},
    },

    'evalution': {
        'model': ['captcha'],
        'captcha': {
            'evaluate': ['hamming'],
            'confuse_matrix': {'predict_path': 'predict.txt',
                               'vector_path': 'tfidfspace_test.dat',
                               'raw_data': 'train_corpus',
                               'image_path': 'bayes.jpg'},

            'f1': {'predict_path': 'predict.txt',
                   'vector_path': 'tfidfspace_test.dat'},

            'roc': {'predict_path': 'predict.txt',
                    'vector_path': 'tfidfspace_test.dat'},

            'kappa': {'predict_path': 'predict.txt',
                    'vector_path': 'tfidfspace_test.dat'},

            'hamming': {'predict_path': 'predict.txt',
                      'vector_path': 'tfidfspace_test.dat'},
        },
    }
}