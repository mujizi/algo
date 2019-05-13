# -*- coding: utf-8 -*-
# @Time    : 2018/10/15 上午10:53
# @Author  : Benqi

# benqi
# SRC_DIR = '/Users/ouhon/PycharmProjects/ai_algo/src/'


CONFIG = {
    'name': 'cnn_keras',
    'task': ['train'],

    'preprocess': {
        'task': ['img2bunch'],

        'img2bunch': {'data_path': 'validation',
                      'bunch_path': 'va_img.dat'
        }
    },




    'train': {
        'model': ['cnn_keras'],

        'cnn_keras': {'data_path': 'img.dat',
                      'model_path': 'cnn_keras.h5',
                      'epochs': 2,
                      'lr': 0.001,
                      'batch_size': 100,
                      'input_shape': (40, 32, 1),
                      'channel_1': 32,
                      'channel_2': 64,
                      'conv_kernel_1': (3, 3),
                      'conv_kernel_2': (3, 3),
                      'dense_1': 256,
                      'dense_2': 10,
                      'pool_size': (2, 2)
                      },
    },

    'predict': {
        'model': ['cnn_keras'],
        'cnn_keras': {'model_path': 'cnn_keras.h5',
                    'data_path': 'va_img.dat',
                    'predict_path': 'cnn_keras.txt'}


    },

    'evalution': {
        'model': ['bayes'],
        'bayes': {
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
