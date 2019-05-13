# -*- coding: utf-8 -*-
# @Time    : 2018/10/15 上午10:53
# @Author  : Benqi

# benqi
# SRC_DIR = '/Users/ouhon/PycharmProjects/ai_algo/src/'


CONFIG = {
    'data_dir':{
        'RAW_DIR': '/Users/ouhon/PycharmProjects/ai_algo/data/raw',
        'PREPROCESS_DIR': '/Users/ouhon/PycharmProjects/ai_algo/data/preprocess',
        'FEATURE_DIR': '/Users/ouhon/PycharmProjects/ai_algo/data/feature',
        'TRAIN_DIR': '/Users/ouhon/PycharmProjects/ai_algo/data/train',
        'PREDICT_DIR': '/Users/ouhon/PycharmProjects/ai_algo/data/predict',
        'EVALUTION_DIR': '/Users/ouhon/PycharmProjects/ai_algo/data/evalution',
        'DICT_DIR': '/Users/ouhon/PycharmProjects/ai_algo/dict'
    },

    'task': ['create_data'],

    'create_data': {
        'task': ['captcha'],
        'captcha': {'data_path': 'captcha/img',
                    'nums': '100'}
    },

    'preprocess': {
        'task': ['csv2bunch'],

        # raw txt file to segment txt file
        'raw2segment': {'data_form': 'sample',
                        'raw_data': 'test_corpus',
                        'pre_data': 'pre_test_data',
                        'stop_word': 'stop_word_1.txt'},

        # segment txt file become tf-idf vector
        'segment2tf': {'data_form': 'sample',
                       'bunch_path': 'test_dataset.dat',
                       'vector_path': 'tfidfspace_test.dat',
                       'visual_vector': 'vocabulary.txt',
                       'tfidfspace_train': 'tfidfspace_train.dat'},

        # h5 raw data to bunch ,bunch like a dictionary
        'h5py2bunch': {'raw_data': 'test_catvnoncat.h5',
                       'bunch_path': 'test_cat_bunch.dat',
                       'x_tag': 'test_set_x',
                       'y_tag': 'test_set_y'
                       },

        # every file transform to utf-8 encode
        'encoder': {
            'raw_data': 'train_catvnoncat.h5',
            'encode_path': 'train_catvnoncat.h5'
        },

        # use pickle to save csv to .dat file
        'csv2dat': {
            'raw_data': 'mnist_csv/linear_submission.csv',
            'pickle_path': 'label_csv.dat',
            'tag': 'ImageId'
        },

        # txt file to bunch file
        'txt2bunch': {
            'raw_data': 'testDigits',
            'bunch_path': 'test_digits.dat'
        },

        # txt segment file to bunch
        'segment2bunch': {'raw_data': '',
                          'bunch_path': '',
                          'data_form': '',
                          'visual_path': ''
                          },

        'csv2bunch': {'csv_path': 'mnist_csv/train.csv',
                      'bunch_path': 'pre/bunch/mnist_bunch.dat'}
    },

    'feature': {
        'task': ['hamming', 'jaccard'],

        # hamming distance
        'hamming': {'data_path': '',
                    'tag': '',
                    'feature_path': ''
        },

        # jaccard similarity coefficien
        'jaccard': {'data_path': '',
                    'tag': '',
                    'feature_path': ''
        },

        # (x-x.min)/(x.max - x.min)
        'normalization': {'data_path': '',
                    'tag': '',
                    'feature_path': ''
        },

        # (x - u)/q
        'standardization': {'data_path': '',
                    'tag': '',
                    'feature_path': ''
        },

        # cos similarity
        'cos_similarity': {'data_path': '',
                    'tag': '',
                    'feature_path': ''
        }
    },

    'train': {
        'model': ['softmax'],
        'bayes': {'data_path': 'tfidfspace_train.dat',
                  'model_path': 'bayes.m',
                  'alpha': 0.01},

        'logistic': {'data_path': 'train_catvnoncat.h5',
                     'model_path': 'logistic.m',
                     'learning_rate': 0.01,
                     'num_iter': 1000},

        'ai_xgboost': {'data_path': 'mnist_csv/train.csv',
                    'model_path': 'ai_xgboost.m',
                    'silent': 0,                        # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
                    'nthread': 4,                       # cpu 线程数 默认最大
                    'learning_rate': 0.3,               # 如同学习率
                    'min_child_weight': 1,              # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言，
                                                        # 假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
                    'max_depth': 10,                    # 构建树的深度，越大越容易过拟合
                    'gamma': 0.1,                       # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
                    'subsample': 1,                     # 随机采样训练样本 训练实例的子采样比
                    'max_delta_step': 0,                # 最大增量步长，我们允许每个树的权重估计。
                    'colsample_bytree': 1,              # 生成树时进行的列采样
                    'reg_lambda': 1,                    # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                    'reg_alpha': 0,                     # L1 正则项参数
                    'scale_pos_weight': 1,              # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
                    'objective': 'reg:logistic',
                                                        # 多分类的问题 指定学习任务和相应的学习目标
                    'n_estimators': 100,                # 树的个数
                    'seed': 1000,                       # 随机种子
                    'eval_metric': 'auc'
                    },

        'softmax': {'data_path': 'train_digits.dat',
                    'model_path': 'softmax.m',
                    'learning_rate': 0.01,
                    'num_iter': 1000},
    },

    'predict': {
        'model': ['softmax'],
        'bayes': {'model_path': 'bayes.m',
                  'data_path': 'tfidfspace_test.dat',
                  'predict_path': 'predict.txt',
                  'format': 'prob'},

        'sample_bayes': {'model_path': 'bayes.m',
                         'data_path': 'sample_tf.dat',
                         'predict_path': 'sample_predict.txt'},

        'logistic': {'model_path': 'logistic.m',
                     'data_path': 'test_catvnoncat.h5',
                     'predict_path': 'logistic_predict.txt'},

        'ai_xgboost': {'model_path': 'ai_xgboost.m',
                       'data_path': 'mnist_csv/test.csv',
                       'predict_path': 'xgboost_predict.txt'},

        'softmax': {'model_path': 'softmax.m',
                    'data_path': 'test_digits.dat',
                    'predict_path': 'softmax.txt'}
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
        'logistic': {
            'evaluate': ['f1'],
            'confuse_matrix': {'predict_path': 'predict.txt',
                       'vector_path': 'tfidfspace_test.dat',
                       'raw_data': 'train_corpus'},

            'f1': {'predict_path': 'logistic_predict.txt',
                   'vector_path': 'test_catvnoncat.h5'},

            'roc': {'model_path': 'bayes.m',
                    'tf_path': 'tfidfspace_test.dat'}
        },
        'ai_xgboost': {
            'evaluate': ['f1'],
            'confuse_matrix': {'predict_path': 'predict.txt',
                               'vector_path': 'tfidfspace_test.dat',
                               'raw_data': 'train_corpus'},

            'f1': {'predict_path': 'xgboost_predict.txt',
                   'vector_path': 'label_csv.dat'},

            'roc': {'model_path': 'bayes.m',
                    'tf_path': 'tfidfspace_test.dat'}
        }

    }
}
