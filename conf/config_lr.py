# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 下午12:51
# @Author  : Benqi

CONFIG = {
    'name': 'lr',
    'task': ['predict'],

    'preprocess': {
        'task': ['h5py2bunch'],
        'h5py2bunch': {
            'raw_data': 'test_catvnoncat.h5',
            'bunch_path': 'test_cat_bunch.dat',
            'x_tag': 'test_set_x',
            'y_tag': 'test_set_y'
        },

    },

    'train': {
        'model': ['logistic'],
        'logistic': {'data_path': 'train_catvnoncat.h5',
                     'model_path': 'logistic.m',
                     'learning_rate': 0.01,
                     'num_iter': 1000},
    },

    'predict': {
        'model': ['logistic'],
        'logistic': {'model_path': 'logistic.m',
                     'data_path': 'test_catvnoncat.h5',
                     'predict_path': 'logistic_predict.txt'},
    },

    'evalution': {
        'model': ['logistic'],
        'bayes': {
            'evaluate': ['f1'],
            'confuse_matrix': {'predict_path': 'lr_predict.txt.txt',
                               'vector_path': 'test_catvnoncat.h5',
                               'raw_data': 'train_corpus',
                               'image_path': 'lr.jpg'},

            'f1': {'predict_path': 'lr_predict.txt.txt',
                   'vector_path': 'test_catvnoncat.h5'},

            'roc': {'predict_path': 'lr_predict.txt.txt',
                    'vector_path': 'test_catvnoncat.h5'},

            'kappa': {'predict_path': 'lr_predict.txt.txt',
                    'vector_path': 'test_catvnoncat.h5'},

            'hamming': {'predict_path': 'lr_predict.txt.txt',
                      'vector_path': 'test_catvnoncat.h5'},
        },
    }
}
