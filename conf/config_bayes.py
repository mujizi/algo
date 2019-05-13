# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 下午12:29
# @Author  : Benqi

CONFIG = {
    'name': 'bayes',
    'task': ['preprocess'],

    'preprocess': {
        'task': ['word2vec'],

        'txt2tf': {'data_form': 'batch',
                   'raw_data': 'test_corpus',
                   'stop_word': 'stop/stop_word_1.txt',
                   'vector_path': 'tfidfspace_test.dat',
                   'visual_vector': 'vocabulary.txt',
                   'tfidfspace_train': 'tfidfspace_train.dat'},

        'raw2segment': {'data_form': 'batch',
                        'raw_data': 'train_corpus',
                        'pre_data': 'pre_train_data',
                        'stop_word': 'stop/stop_word_1.txt'},

        'segment2bunch': {'raw_data': 'pre_test_data',
                          'bunch_path': 'test_dataset.dat',
                          'visual_path': 'visualization.txt'},

        'segment2tf': {'data_form': 'batch',
                       'bunch_path': 'test_dataset.dat',
                       'vector_path': 'tfidfspace_test.dat',
                       'visual_vector': 'vocabulary.txt',
                       'tfidfspace_train': 'tfidfspace_train.dat'},

        'txt2word': {'pre_data': 'C4-Literature',
                     'word_path': 'c4.txt'},

        'word2vec': {'pre_data': 'c4.txt',
                     'vector_path': 'c4.npy'},
    },

    'train': {
        'model': ['bayes'],

        'bayes': {'data_path': 'tfidfspace_train.dat',
                  'model_path': 'bayes.m',
                  'alpha': 0.01}
    },

    'predict': {
        'model': ['bayes'],
        'bayes': {'model_path': 'bayes.m',
                  'data_path': 'tfidfspace_test.dat',
                  'predict_path': 'bayes_predict.txt',
                  'format': 'pr'},
    },

    'evalution': {
        'model': ['bayes'],
        'bayes': {
            'evaluate': ['f1'],
            'confuse_matrix': {'predict_path': 'y_pred/bayes_predict.txt',
                               'vector_path': 'id/tfidfspace_test.dat',
                               'raw_data': 'corpus/train_corpus',
                               'image_path': 'visualization/bayes.jpg'},

            'f1': {'predict_path': 'y_pred/bayes_predict.txt',
                   'vector_path': 'id/tfidfspace_test.dat'},

            'roc': {'predict_path': 'y_pred/bayes_predict.txt',
                    'vector_path': 'id/tfidfspace_test.dat'},

            'kappa': {'predict_path': 'y_pred/bayes_predict.txt',
                    'vector_path': 'id/tfidfspace_test.dat'},

            'hamming': {'predict_path': 'y_pred/bayes_predict.txt',
                      'vector_path': 'id/tfidfspace_test.dat'},
        },
    }
}