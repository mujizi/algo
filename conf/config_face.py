# -*- coding: utf-8 -*-
# @Time    : 2018/12/14 下午5:17
# @Author  : Benqi
CONFIG = {
    'name': 'face',
    'task': ['preprocess'],

    'preprocess': {
        'task': ['img_reshape'],
        'img_reshape': {
            'raw_data': '90.jpg',
            'pre_data': '90.jpg',
            'pickle_path': 'data.pkl'
        },

    },

    'train': {
        'model': ['ai_face'],

        'ai_face': {'data_path': 'data.pkl',
                    'model_path': 'model.h5',
                    'input_shape': (64, 64, 3)}
    },

    'predict': {
        'model': ['ai_face'],
        'ai_face': {'model_path': 'model.h5',
                    'data_path': 'Alan_Greenspan_0002.jpg',
                    'format': 'sample',
                    'input_shape': (-1, 64, 64, 3),
                    'predict_path': 'face.txt'},
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
