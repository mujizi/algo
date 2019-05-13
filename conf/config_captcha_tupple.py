# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 下午4:17
# @Author  : Benqi

CONFIG = {
    'name': 'captcha',
    'task': ['predict'],

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
                    'model_path': 'model',
                    'input_shape': (160, 60, 3),
                    'sample_nums': 100,
                    'train_batch_size': 5,
                    'dev_batch_size': 2,
                    'test_batch_size': 2,
                    'learning_rate': 0.001,
                    'epoch_nums': 2,
                    'epochs_per_dev': 1,
                    'epochs_per_save': 2,
                    'steps_per_print': 2,
                    'keep_prob': 0.5,
                    'vocab': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                    'captcha_length': 4,
                    }
    },

    'predict': {
        'model': ['captcha'],
        'captcha': {'model_path': 'model.meta',
                    'data_path': 'beii.jpg',
                    'format': 'sample',
                    'input_shape': (-1, 160, 60, 3),
                    'vocab': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                    'captcha_length': 4,
                    'keep_prob': 0.5,
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