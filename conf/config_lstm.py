# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 下午3:02
# @Author  : Benqi
CONFIG = {
    'name':'lstm',
    'task': ['predict'],

    'preprocess': {
        'task': ['series2supervised'],
        'data_extract': {'raw_data': 'pollution_3.csv',
                         'csv_path': 'pollution_4.csv',
                         'task': 'normalization',
                         'col_name': 'wnd_dir',
                         'col_nums': 4,
                         'row_name': 'a',
                         'values': 'a'
                         },

        'series2supervised': {'raw_data': 'pollution_4.csv',
                              'csv_path': 'pollution_5.csv',
                              'n_in': 1,
                              'n_out': 1,
                              'dropna': True}
    },

    'train': {
        'model': ['ai_lstm'],
        'ai_lstm': {'data_path': 'pollution_5.csv',
                    'model_path': 'ai_lstm.h5'},
    },

    'predict': {
        'model': ['ai_lstm'],
        'ai_lstm': {'model_path': 'ai_lstm.h5',
                    'data_path': 'pollution_5.csv',
                    'predict_path': 'y_pred.csv'}
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