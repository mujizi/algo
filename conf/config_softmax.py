# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 下午12:51
# @Author  : Benqi

CONFIG = {
    'name':'softmax',
    'task': ['train'],

    'preprocess': {
        'task': ['txt2bunch'],
        'txt2bunch': {'raw_data': 'testDigits',
                      'bunch_path': 'test_digits.dat'},
    },

    'train': {
        'model': ['softmax'],
        'softmax': {'data_path': 'train_digits.dat',
                    'model_path': 'softmax.m',
                    'learning_rate': 0.01,
                    'num_iter': 1000},
    },

    'predict': {
        'model': ['softmax'],
        'softmax': {'model_path': 'softmax.m.meta',
                    'data_path': 'test_digits.dat',
                    'predict_path': 'softmax.txt'}
    },

    'evalution': {
        'model': ['softmax'],
        'bayes': {
            'evaluate': ['f1'],
            'confuse_matrix': {'predict_path': 'softmax.txt',
                               'raw_data': 'train_corpus',
                               'image_path': 'softmax.jpg'},

            'f1': {'predict_path': 'softmax.txt',
                   'vector_path': 'test_digits.dat'},

            'roc': {'predict_path': 'softmax.txt',
                    'vector_path': 'test_digits.dat'},

            'kappa': {'predict_path': 'softmax.txt',
                    'vector_path': 'test_digits.dat'},

            'hamming': {'predict_path': 'softmax.txt',
                      'vector_path': 'test_digits.dat'},
        },
    }
}