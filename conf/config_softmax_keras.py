# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 下午12:51
# @Author  : Benqi

CONFIG = {
    'name': 'softmax_keras',
    'task': ['predict'],

    'preprocess': {
        'task': ['txt2bunch'],
        'txt2bunch': {'raw_data': 'trainingDigits',
                      'bunch_path': 'train_digits.dat'},
    },

    'train': {
        'model': ['softmax_keras'],
        'softmax_keras': {'data_path': 'train_digits.dat',
                          'model_path': 'softmax_keras.h5'},
    },

    'predict': {
        'model': ['softmax_keras'],
        'softmax_keras': {'model_path': 'softmax_keras.h5',
                    'data_path': 'test_digits.dat',
                    'predict_path': 'softmax_keras.txt'}
    },

    'evalution': {
        'model': ['softmax_keras'],
        'bayes': {
            'evaluate': ['f1'],
            'confuse_matrix': {'predict_path': 'softmax_keras.txt',
                               'raw_data': 'train_corpus',
                               'image_path': 'softmax_keras.jpg'},

            'f1': {'predict_path': 'softmax_keras.txt',
                   'vector_path': 'test_digits.dat'},

            'roc': {'predict_path': 'softmax_keras.txt',
                    'vector_path': 'test_digits.dat'},

            'kappa': {'predict_path': 'softmax_keras.txt',
                    'vector_path': 'test_digits.dat'},

            'hamming': {'predict_path': 'softmax_keras.txt',
                      'vector_path': 'test_digits.dat'},
        },
    }
}