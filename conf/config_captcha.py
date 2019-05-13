# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 下午7:49
# @Author  : Benqi

CONFIG = {
    'name': 'captcha',
    'task': ['train'],

    'create_data': {
        'task': ['captcha'],
        'captcha': {'data_path': '',
                    'pickle_path': 'data.pkl',
                    'vocab': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                    'captcha_length': 4,
                    'nums': '100'}
    },

    'train': {
        'model': ['captcha'],
        'captcha': {'data_path': 'data.pkl',
                    'model_path': '666',
                    'input_shape': (160, 60, 3),
                    'sample_nums': 100,
                    'batch_size': 50,
                    'learning_rate': 0.001,
                    'epoch_nums': 1000,
                    'keep_prob': 1,
                    'vocab': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                    'captcha_length': 4,
                    }
    },

    'predict': {
        'model': ['captcha'],
        'captcha': {'model_path': '666.meta',
                    'data_path': 'data.pkl',
                    'input_shape': (-1, 160, 60, 3),
                    'vocab': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
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