# -*- coding:utf-8 -*-

import sys
sys.path.append(r"..")

import logging
from conf.config_base import CONFIG_BASE

class Config:
    dic_config = {}
    name = ''
    def init(self, dic_command, dic_config):
        self.dic_config.update(CONFIG_BASE)
        self.dic_config.update(dic_config)
        self.dic_config.update(dic_command)
        if 'name' in self.dic_config:
            self.name = self.dic_config['name']

    def set_data_dir(self):
        if 'DATA_DIR' not in self.dic_config:
            return

        dic_data_dir = self.dic_config['DATA_DIR']
        if self.name:
            dic_data_dir['RAW_DIR'] = '%s/%s/%s' % (dic_data_dir['ROOT_DIR'], self.name, dic_data_dir['RAW_DIR'])
        else:
            dic_data_dir['RAW_DIR'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['RAW_DIR'])

        if self.name:
            dic_data_dir['CREATE_DIR'] = '%s/%s/%s' % (dic_data_dir['ROOT_DIR'], self.name, dic_data_dir['CREATE_DIR'])
        else:
            dic_data_dir['CREATE_DIR'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['CREATE_DIR'])

        if self.name:
            dic_data_dir['PREPROCESS_DIR'] = '%s/%s/%s' % (dic_data_dir['ROOT_DIR'], self.name, dic_data_dir['PREPROCESS_DIR'])
        else:
            dic_data_dir['PREPROCESS_DIR'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['PREPROCESS_DIR'])

        if self.name:
            dic_data_dir['FEATURE_DIR'] = '%s/%s/%s' % (dic_data_dir['ROOT_DIR'], self.name, dic_data_dir['FEATURE_DIR'])
        else:
            dic_data_dir['FEATURE_DIR'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['FEATURE_DIR'])

        if self.name:
            dic_data_dir['TRAIN_DIR'] = '%s/%s/%s' % (dic_data_dir['ROOT_DIR'], self.name, dic_data_dir['TRAIN_DIR'])
        else:
            dic_data_dir['TRAIN_DIR'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['TRAIN_DIR'])

        if self.name:
            dic_data_dir['PREDICT_DIR'] = '%s/%s/%s' % (dic_data_dir['ROOT_DIR'], self.name, dic_data_dir['PREDICT_DIR'])
        else:
            dic_data_dir['PREDICT_DIR'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['PREDICT_DIR'])

        if self.name:
            dic_data_dir['EVALUTION_DIR'] = '%s/%s/%s' % (dic_data_dir['ROOT_DIR'], self.name, dic_data_dir['EVALUTION_DIR'])
        else:
            dic_data_dir['EVALUTION_DIR'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['EVALUTION_DIR'])

    def create_data(self):
        if 'task' not in self.dic_config['create_data']:
            logging.warning("task no set")

        dic_data_dir = self.dic_config['DATA_DIR']
        if 'captcha' in self.dic_config['create_data']:
            dic_captcha = self.dic_config['create_data']['captcha']

            if 'data_path' not in dic_captcha:
                logging.error('data_path no set')
                return False
            dic_captcha['data_path'] = '%s/%s' % (dic_data_dir['CREATE_DIR'], dic_captcha['data_path'])

            if 'pickle_path' not in dic_captcha:
                logging.error('pickle_path no set')
                return False
            dic_captcha['pickle_path'] = '%s/%s' % (dic_data_dir['CREATE_DIR'], dic_captcha['pickle_path'])

        return True

    def preprocess(self):
        if 'task' not in self.dic_config['preprocess']:
            logging.warning("task no set")

        dic_data_dir = self.dic_config['DATA_DIR']
        if 'raw2segment' in self.dic_config['preprocess']:
            dic_raw2segment = self.dic_config['preprocess']['raw2segment']

            if 'raw_data' not in dic_raw2segment:
                logging.error('raw_data no set')
                return False
            dic_raw2segment['raw_data'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_raw2segment['raw_data'])

            if 'pre_data' not in dic_raw2segment:
                logging.error('pre_data no set')
                return False
            dic_raw2segment['pre_data'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_raw2segment['pre_data'])

            if 'stop_word' not in dic_raw2segment:
                logging.error('stop_word no set')
                return False
            dic_raw2segment['stop_word'] = '%s/%s' % (self.dic_config['DICT_DIR'], dic_raw2segment['stop_word'])

        if 'segment2tf' in self.dic_config['preprocess']:
            dic_segment2tf = self.dic_config['preprocess']['segment2tf']

            if 'bunch_path' not in dic_segment2tf:
                logging.error('bunch_path no set')
                return False
            dic_segment2tf['bunch_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_segment2tf['bunch_path'])

            if 'vector_path' not in dic_segment2tf:
                logging.error('vector_path no set')
                return False
            dic_segment2tf['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_segment2tf['vector_path'])

            if 'visual_vector' not in dic_segment2tf:
                logging.error('stop_word no set')
                return False
            dic_segment2tf['visual_vector'] = '%s/%s' % (dic_data_dir['EVALUTION_DIR'], dic_segment2tf['visual_vector'])

            if 'tfidfspace_train' in dic_segment2tf:
                dic_segment2tf['tfidfspace_train'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_segment2tf['tfidfspace_train'])

        if 'txt2tf' in self.dic_config['preprocess']:
            dic_txt2tf = self.dic_config['preprocess']['txt2tf']

            if 'raw_data' not in dic_txt2tf:
                logging.error('raw_data no set')
                return False
            dic_txt2tf['raw_data'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_txt2tf['raw_data'])

            if 'vector_path' not in dic_txt2tf:
                logging.error('vector_path no set')
                return False
            dic_txt2tf['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_txt2tf['vector_path'])

            if 'visual_vector' not in dic_txt2tf:
                logging.error('stop_word no set')
                return False
            dic_txt2tf['visual_vector'] = '%s/%s' % (dic_data_dir['EVALUTION_DIR'], dic_txt2tf['visual_vector'])

            if 'stop_word' not in dic_txt2tf:
                logging.error('stop_word no set')
                return False
            dic_txt2tf['stop_word'] = '%s/%s' % (self.dic_config['DICT_DIR'], dic_txt2tf['stop_word'])

            if 'tfidfspace_train' in dic_txt2tf:
                dic_txt2tf['tfidfspace_train'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_txt2tf['tfidfspace_train'])

        if 'h5py2bunch' in self.dic_config['preprocess']:
            dic_h5py2bunch = self.dic_config['preprocess']['h5py2bunch']

            if 'raw_data' not in dic_h5py2bunch:
                logging.error('raw_data no set')
                return False
            dic_h5py2bunch['raw_data'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_h5py2bunch['raw_data'])

            if 'bunch_path' not in dic_h5py2bunch:
                logging.error('bunch_path no set')
                return False
            dic_h5py2bunch['bunch_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_h5py2bunch['bunch_path'])

        if 'encoder' in self.dic_config['preprocess']:
            dic_encoder = self.dic_config['preprocess']['encoder']

            if 'raw_data' not in dic_encoder:
                logging.error('raw_data no set')
                return False
            dic_encoder['raw_data'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_encoder['raw_data'])

            if 'encode_path' not in dic_encoder:
                logging.error('encode_path no set')
                return False
            dic_encoder['encode_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_encoder['encode_path'])


        if 'csv2dat' in self.dic_config['preprocess']:
            dic_csv2dat = self.dic_config['preprocess']['csv2dat']

            if 'raw_data' not in dic_csv2dat:
                logging.error('raw_data no set')
                return False
            dic_csv2dat['raw_data'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_csv2dat['raw_data'])

            if 'pickle_path' not in dic_csv2dat:
                logging.error('pickle_path no set')
                return False
            dic_csv2dat['pickle_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_csv2dat['pickle_path'])


        if 'segment2bunch' in self.dic_config['preprocess']:
            dic_segment2bunch = self.dic_config['preprocess']['segment2bunch']

            if 'raw_data' not in dic_segment2bunch:
                logging.error('raw_data no set')
                return False
            dic_segment2bunch['raw_data'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_segment2bunch['raw_data'])

            if 'bunch_path' not in dic_segment2bunch:
                logging.error('bunch_path no set')
                return False
            dic_segment2bunch['bunch_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_segment2bunch['bunch_path'])

            if 'visual_path' not in dic_segment2bunch:
                logging.error('visual_path no set')
                return False
            dic_segment2bunch['visual_path'] = '%s/%s' % (dic_data_dir['EVALUTION_DIR'], dic_segment2bunch['visual_path'])


        if 'txt2bunch' in self.dic_config['preprocess']:
            dic_txt2bunch = self.dic_config['preprocess']['txt2bunch']

            if 'raw_data' not in dic_txt2bunch:
                logging.error('raw_data no set')
                return False
            dic_txt2bunch['raw_data'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_txt2bunch['raw_data'])

            if 'bunch_path' not in dic_txt2bunch:
                logging.error('bunch_path no set')
                return False
            dic_txt2bunch['bunch_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_txt2bunch['bunch_path'])


        if 'bunch2csv' in self.dic_config['preprocess']:
            dic_bunch2csv = self.dic_config['preprocess']['img_reshape']

            if 'raw_data' not in dic_bunch2csv:
                logging.error('raw_data no set')
                return False
            dic_bunch2csv['raw_data'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_bunch2csv['raw_data'])

            if 'csv_path' not in dic_bunch2csv:
                logging.error('csv_path no set')
                return False
            dic_bunch2csv['csv_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_bunch2csv['csv_path'])


        if 'package_dataset' in self.dic_config['preprocess']:
            dic_package_dataset = self.dic_config['preprocess']['img_reshape']

            if 'csv_path' not in dic_package_dataset:
                logging.error('csv_path no set')
                return False
            dic_package_dataset['csv_path'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_package_dataset['csv_path'])


        if 'img_reshape' in self.dic_config['preprocess']:
            dic_img_reshape = self.dic_config['preprocess']['img_reshape']

            if 'raw_data' not in dic_img_reshape:
                logging.error('raw_data no set')
                return False
            dic_img_reshape['raw_data'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_img_reshape['raw_data'])

            if 'pre_data' not in dic_img_reshape:
                logging.error('pre_data no set')
                return False
            dic_img_reshape['pre_data'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_img_reshape['pre_data'])

            if 'pickle_path' not in dic_img_reshape:
                logging.error('pickle_path no set')
                return False
            dic_img_reshape['pickle_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_img_reshape['pickle_path'])

        if 'img2bunch' in self.dic_config['preprocess']:
            dic_img2bunch = self.dic_config['preprocess']['img2bunch']

            if 'data_path' not in dic_img2bunch:
                logging.error('data_path no set')
                return False
            dic_img2bunch['data_path'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_img2bunch['data_path'])

            if 'bunch_path' not in dic_img2bunch:
                logging.error('bunch_path no set')
                return False
            dic_img2bunch['bunch_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_img2bunch['bunch_path'])

        if 'csv2bunch' in self.dic_config['preprocess']:
            dic_csv2bunch = self.dic_config['preprocess']['csv2bunch']

            if 'csv_path' not in dic_csv2bunch:
                logging.error('csv_path no set')
                return False
            dic_csv2bunch['csv_path'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_csv2bunch['csv_path'])

            if 'bunch_path' not in dic_csv2bunch:
                logging.error('bunch_path no set')
                return False
            dic_csv2bunch['bunch_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_csv2bunch['bunch_path'])

        if 'matrix2sparse' in self.dic_config['preprocess']:
            dic_matrix2sparse = self.dic_config['preprocess']['matrix2sparse']

            if 'data_path' not in dic_matrix2sparse:
                logging.error('data_path no set')
                return False
            dic_matrix2sparse['data_path'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_matrix2sparse['data_path'])

            if 'sparse_path' not in dic_matrix2sparse:
                logging.error('sparse_path no set')
                return False
            dic_matrix2sparse['sparse_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_matrix2sparse['sparse_path'])

        if 'data_extract' in self.dic_config['preprocess']:
            dic_data_extract = self.dic_config['preprocess']['data_extract']

            if 'raw_data' not in dic_data_extract:
                logging.error('raw_data no set')
                return False
            dic_data_extract['raw_data'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_data_extract['raw_data'])

            if 'csv_path' not in dic_data_extract:
                logging.error('csv_path no set')
                return False
            dic_data_extract['csv_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_data_extract['csv_path'])

        if 'series2supervised' in self.dic_config['preprocess']:
            dic_series2supervised = self.dic_config['preprocess']['series2supervised']

            if 'raw_data' not in dic_series2supervised:
                logging.error('raw_data no set')
                return False
            dic_series2supervised['raw_data'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_series2supervised['raw_data'])

            if 'csv_path' not in dic_series2supervised:
                logging.error('csv_path no set')
                return False
            dic_series2supervised['csv_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_series2supervised['csv_path'])

        if 'txt2word' in self.dic_config['preprocess']:
            dic_txt2word = self.dic_config['preprocess']['txt2word']

            if 'pre_data' not in dic_txt2word:
                logging.error('raw_data no set')
                return False
            dic_txt2word['pre_data'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_txt2word['pre_data'])

            if 'word_path' not in dic_txt2word:
                logging.error('word_path no set')
                return False
            dic_txt2word['word_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_txt2word['word_path'])

        if 'word2vec' in self.dic_config['preprocess']:
            dic_word2vec = self.dic_config['preprocess']['word2vec']

            if 'pre_data' not in dic_word2vec:
                logging.error('raw_data no set')
                return False
            dic_word2vec['pre_data'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_word2vec['pre_data'])

            if 'vector_path' not in dic_word2vec:
                logging.error('vector_path no set')
                return False
            dic_word2vec['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_word2vec['vector_path'])

        return True


    def feature(self):
        if 'task' not in self.dic_config['feature']:
            logging.warning("task not set")

        if 'segment2libsvm' in self.dic_config['feature']:
            if 'pre_data' not in self.dic_config['feature']['segment2libsvm']:
                logging.error("pre_data no set")
                return False

        if 'tf_idf' in self.dic_config['feature']:
            if 'data_path' not in self.dic_config['feature']['tf_idf']:
                logging.error("data_path no set")
                return False

            if 'keyword_path' not in self.dic_config['feature']['tf_idf']:
                logging.error("keyword_path no set")
                return False

        return True


    def train(self):
        if 'model' not in self.dic_config['train']:
            logging.warning('model not set')

        dic_data_dir = self.dic_config['DATA_DIR']
        if 'bayes' in self.dic_config['train']:
            dic_bayes = self.dic_config['train']['bayes']

            if 'data_path' not in dic_bayes:
                logging.error('data_path no set')
                return False
            dic_bayes['data_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_bayes['data_path'])

            if 'model_path' not in dic_bayes:
                logging.error('model_path no set')
                return False
            dic_bayes['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_bayes['model_path'])

        if 'logistic' in self.dic_config['train']:
            dic_logistic = self.dic_config['train']['logistic']

            if 'data_path' not in dic_logistic:
                logging.error('data_path no set')
                return False
            dic_logistic['data_path'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_logistic['data_path'])

            if 'model_path' not in dic_logistic:
                logging.error('model_path no set')
                return False
            dic_logistic['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_logistic['model_path'])

        if 'softmax' in self.dic_config['train']:
            dic_softmax = self.dic_config['train']['softmax']

            if 'data_path' not in dic_softmax:
                logging.error('data_path no set')
                return False
            dic_softmax['data_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_softmax['data_path'])

            if 'model_path' not in dic_softmax:
                logging.error('model_path no set')
                return False
            dic_softmax['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_softmax['model_path'])

        if 'softmax_keras' in self.dic_config['train']:
            dic_softmax_keras = self.dic_config['train']['softmax_keras']

            if 'data_path' not in dic_softmax_keras:
                logging.error('data_path no set')
                return False
            dic_softmax_keras['data_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_softmax_keras['data_path'])

            if 'model_path' not in dic_softmax_keras:
                logging.error('model_path no set')
                return False
            dic_softmax_keras['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_softmax_keras['model_path'])

        if 'captcha' in self.dic_config['train']:
            dic_captcha = self.dic_config['train']['captcha']

            if 'data_path' not in dic_captcha:
                logging.error('data_path no set')
                return False
            dic_captcha['data_path'] = '%s/%s' % (dic_data_dir['CREATE_DIR'], dic_captcha['data_path'])

            if 'model_path' not in dic_captcha:
                logging.error('model_path no set')
                return False
            dic_captcha['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_captcha['model_path'])

        if 'ai_xgboost' in self.dic_config['train']:
            dic_ai_xgboost = self.dic_config['train']['ai_xgboost']

            if 'data_path' not in dic_ai_xgboost:
                logging.error('data_path no set')
                return False
            dic_ai_xgboost['data_path'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_ai_xgboost['data_path'])

            if 'sparse_path' not in dic_ai_xgboost:
                logging.error('sparse_path no set')
                return False
            dic_ai_xgboost['sparse_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_ai_xgboost['sparse_path'])

            if 'model_path' not in dic_ai_xgboost:
                logging.error('model_path no set')
                return False
            dic_ai_xgboost['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_ai_xgboost['model_path'])


        if 'cnn_keras' in self.dic_config['train']:
            dic_cnn_keras = self.dic_config['train']['cnn_keras']

            if 'data_path' not in dic_cnn_keras:
                logging.error('data_path no set')
                return False
            dic_cnn_keras['data_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_cnn_keras['data_path'])

            if 'model_path' not in dic_cnn_keras:
                logging.error('model_path no set')
                return False
            dic_cnn_keras['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_cnn_keras['model_path'])

        if 'ai_lstm' in self.dic_config['train']:
            dic_ai_lstm = self.dic_config['train']['ai_lstm']

            if 'data_path' not in dic_ai_lstm:
                logging.error('data_path no set')
                return False
            dic_ai_lstm['data_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_ai_lstm['data_path'])

            if 'model_path' not in dic_ai_lstm:
                logging.error('model_path no set')
                return False
            dic_ai_lstm['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_ai_lstm['model_path'])

        if 'ai_face' in self.dic_config['train']:
            dic_ai_face = self.dic_config['train']['ai_face']

            if 'data_path' not in dic_ai_face:
                logging.error('data_path no set')
                return False
            dic_ai_face['data_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_ai_face['data_path'])

            if 'model_path' not in dic_ai_face:
                logging.error('model_path no set')
                return False
            dic_ai_face['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_ai_face['model_path'])

        return True


    def predict(self):
        if 'model' not in self.dic_config['predict']:
            logging.warning("model not set")

        dic_data_dir = self.dic_config['DATA_DIR']
        if 'bayes' in self.dic_config['predict']:
            dic_bayes = self.dic_config['predict']['bayes']

            if 'predict_path' not in dic_bayes:
                logging.error("predict_path no set")
                return False
            dic_bayes['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_bayes['predict_path'])

            ##########
            if 'data_path' not in dic_bayes:
                logging.error("data_path no set")
                return False
            dic_bayes['data_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_bayes['data_path'])

            ############
            if 'model_path' not in dic_bayes:
                logging.error("data no set")
                return False
            dic_bayes['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_bayes['model_path'])

        if 'sample_bayes' in self.dic_config['predict']:
            dic_sample_bayes = self.dic_config['predict']['sample_bayes']

            if 'predict_path' not in dic_sample_bayes:
                logging.error("predict_path no set")
                return False
            dic_sample_bayes['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_sample_bayes['predict_path'])

            if 'data_path' not in dic_sample_bayes:
                logging.error("data_path no set")
                return False
            dic_sample_bayes['data_path'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_sample_bayes['data_path'])

            if 'model_path' not in dic_sample_bayes:
                logging.error("data no set")
                return False
            dic_sample_bayes['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_sample_bayes['model_path'])

        if 'logistic' in self.dic_config['predict']:
            dic_logistic = self.dic_config['predict']['logistic']

            if 'predict_path' not in dic_logistic:
                logging.error("predict_path no set")
                return False
            dic_logistic['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_logistic['predict_path'])

            if 'data_path' not in dic_logistic:
                logging.error("data_path no set")
                return False
            dic_logistic['data_path'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_logistic['data_path'])

            if 'model_path' not in dic_logistic:
                logging.error("data no set")
                return False
            dic_logistic['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_logistic['model_path'])

        if 'ai_xgboost' in self.dic_config['predict']:
            dic_ai_xgboost = self.dic_config['predict']['ai_xgboost']

            if 'predict_path' not in dic_ai_xgboost:
                logging.error("predict_path no set")
                return False
            dic_ai_xgboost['predict_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_ai_xgboost['predict_path'])

            if 'data_path' not in dic_ai_xgboost:
                logging.error("data_path no set")
                return False
            dic_ai_xgboost['data_path'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_ai_xgboost['data_path'])

            if 'model_path' not in dic_ai_xgboost:
                logging.error("data no set")
                return False
            dic_ai_xgboost['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_ai_xgboost['model_path'])

        if 'softmax' in self.dic_config['predict']:
            dic_softmax = self.dic_config['predict']['softmax']

            if 'predict_path' not in dic_softmax:
                logging.error("predict_path no set")
                return False
            dic_softmax['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_softmax['predict_path'])

            if 'data_path' not in dic_softmax:
                logging.error("data_path no set")
                return False
            dic_softmax['data_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_softmax['data_path'])

            if 'model_path' not in dic_softmax:
                logging.error("data no set")
                return False
            dic_softmax['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_softmax['model_path'])

        if 'softmax_keras' in self.dic_config['predict']:
            dic_softmax_keras = self.dic_config['predict']['softmax_keras']

            if 'predict_path' not in dic_softmax_keras:
                logging.error("predict_path no set")
                return False
            dic_softmax_keras['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_softmax_keras['predict_path'])

            if 'data_path' not in dic_softmax_keras:
                logging.error("data_path no set")
                return False
            dic_softmax_keras['data_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_softmax_keras['data_path'])

            if 'model_path' not in dic_softmax_keras:
                logging.error("data no set")
                return False
            dic_softmax_keras['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_softmax_keras['model_path'])


        if 'cnn_keras' in self.dic_config['predict']:
            dic_cnn_keras = self.dic_config['predict']['cnn_keras']

            if 'data_path' not in dic_cnn_keras:
                logging.error('data_path no set')
                return False
            dic_cnn_keras['data_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_cnn_keras['data_path'])

            if 'model_path' not in dic_cnn_keras:
                logging.error('model_path no set')
                return False
            dic_cnn_keras['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_cnn_keras['model_path'])

            if 'predict_path' not in dic_cnn_keras:
                logging.error('predict_path no set')
                return False
            dic_cnn_keras['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_cnn_keras['predict_path'])

        if 'captcha' in self.dic_config['predict']:
            dic_captcha = self.dic_config['predict']['captcha']

            if 'data_path' not in dic_captcha:
                logging.error('data_path no set')
                return False
            dic_captcha['data_path'] = '%s/%s' % (dic_data_dir['CREATE_DIR'], dic_captcha['data_path'])

            if 'model_path' not in dic_captcha:
                logging.error('model_path no set')
                return False
            dic_captcha['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_captcha['model_path'])

            if 'predict_path' not in dic_captcha:
                logging.error('predict_path no set')
                return False
            dic_captcha['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_captcha['predict_path'])

        if 'captcha_sample' in self.dic_config['predict']:
            dic_captcha_sample = self.dic_config['predict']['captcha_sample']

            if 'data_path' not in dic_captcha_sample:
                logging.error('data_path no set')
                return False
            dic_captcha_sample['data_path'] = '%s/%s' % (dic_data_dir['CREATE_DIR'], dic_captcha_sample['data_path'])

            if 'model_path' not in dic_captcha_sample:
                logging.error('model_path no set')
                return False
            dic_captcha_sample['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_captcha_sample['model_path'])

            if 'predict_path' not in dic_captcha_sample:
                logging.error('predict_path no set')
                return False
            dic_captcha_sample['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_captcha_sample['predict_path'])

        if 'ai_lstm' in self.dic_config['predict']:
            dic_ai_lstm = self.dic_config['predict']['ai_lstm']

            if 'data_path' not in dic_ai_lstm:
                logging.error('data_path no set')
                return False
            dic_ai_lstm['data_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_ai_lstm['data_path'])

            if 'model_path' not in dic_ai_lstm:
                logging.error('model_path no set')
                return False
            dic_ai_lstm['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_ai_lstm['model_path'])

            if 'predict_path' not in dic_ai_lstm:
                logging.error('predict_path no set')
                return False
            dic_ai_lstm['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_ai_lstm['predict_path'])

        if 'ai_face' in self.dic_config['predict']:
            dic_ai_face = self.dic_config['predict']['ai_face']

            if 'data_path' not in dic_ai_face:
                logging.error('data_path no set')
                return False
            dic_ai_face['data_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_ai_face['data_path'])

            if 'model_path' not in dic_ai_face:
                logging.error('model_path no set')
                return False
            dic_ai_face['model_path'] = '%s/%s' % (dic_data_dir['TRAIN_DIR'], dic_ai_face['model_path'])

            if 'predict_path' not in dic_ai_face:
                logging.error('predict_path no set')
                return False
            dic_ai_face['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_ai_face['predict_path'])

        return True

    def evalution(self):
        if 'model' not in self.dic_config['evalution']:
            logging.warning("task not set")

        dic_data_dir = self.dic_config['DATA_DIR']
        if 'bayes' in self.dic_config['evalution']['model']:
            if 'confuse_matrix' in self.dic_config['evalution']['bayes']['evaluate']:
                dic_confuse_matrix = self.dic_config['evalution']['bayes']

                if 'predict_path' not in dic_confuse_matrix['confuse_matrix']:
                    logging.error("predict_path no set")
                    return False
                dic_confuse_matrix['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_confuse_matrix['predict_path'])

                if 'raw_data' not in dic_confuse_matrix['confuse_matrix']:
                    logging.error("raw_data no set")
                    return False
                dic_confuse_matrix['raw_data'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_confuse_matrix['raw_data'])

                if 'vector_path' not in dic_confuse_matrix['confuse_matrix']:
                    logging.error("vector no set")
                    return False
                dic_confuse_matrix['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_confuse_matrix['vector_path'])

                if 'image_path' not in dic_confuse_matrix['confuse_matrix']:
                    logging.error("image_path no set")
                    return False
                dic_confuse_matrix['image_path'] = '%s/%s' % (dic_data_dir['EVALUTION_DIR'], dic_confuse_matrix['image_path'])


            if 'roc' in self.dic_config['evalution']['bayes']['evaluate']:
                dic_roc = self.dic_config['evalution']['bayes']

                if 'predict_path' not in dic_roc['roc']:
                    logging.error("predict_path no set")
                    return False
                dic_roc['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_roc['predict_path'])

                if 'vector_path' not in dic_roc['roc']:
                    logging.error("vector no set")
                    return False
                dic_roc['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_roc['vector_path'])

                if 'image_path' not in dic_roc['roc']:
                    logging.error("image_path no set")
                    return False
                dic_roc['image_path'] = '%s/%s' % (dic_data_dir['EVALUTION_DIR'], dic_roc['image_path'])


            if 'f1' in self.dic_config['evalution']['bayes']['evaluate']:
                dic_f1 = self.dic_config['evalution']['bayes']

                if 'predict_path' not in dic_f1['f1']:
                    logging.error("predict_path no set")
                    return False
                dic_f1['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_f1['predict_path'])

                if 'vector_path' not in dic_f1['f1']:
                    logging.error("vector no set")
                    return False
                dic_f1['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_f1['vector_path'])


            if 'kappa' in self.dic_config['evalution']['bayes']['evaluate']:
                dic_kappa = self.dic_config['evalution']['bayes']

                if 'predict_path' not in dic_kappa['kappa']:
                    logging.error("predict_path no set")
                    return False
                dic_kappa['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_kappa['predict_path'])

                if 'vector_path' not in dic_kappa['kappa']:
                    logging.error("vector no set")
                    return False
                dic_kappa['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_kappa['vector_path'])


            if 'hamming' in self.dic_config['evalution']['bayes']['evaluate']:
                dic_hamming = self.dic_config['evalution']['bayes']

                if 'predict_path' not in dic_hamming['hamming']:
                    logging.error("predict_path no set")
                    return False
                dic_hamming['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_hamming['predict_path'])

                if 'vector_path' not in dic_hamming['hamming']:
                    logging.error("vector no set")
                    return False
                dic_hamming['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_hamming['vector_path'])

            return True


        if 'logistic' in self.dic_config['evalution']['model']:

            if 'confuse_matrix' in self.dic_config['evalution']['logistic']['evaluate']:
                dic_confuse_matrix = self.dic_config['evalution']['logistic']

                if 'predict_path' not in dic_confuse_matrix['confuse_matrix']:
                    logging.error("predict_path no set")
                    return False
                dic_confuse_matrix['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_confuse_matrix['predict_path'])

                if 'raw_data' not in dic_confuse_matrix['confuse_matrix']:
                    logging.error("raw_data no set")
                    return False
                dic_confuse_matrix['raw_data'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_confuse_matrix['raw_data'])

                if 'vector_path' not in dic_confuse_matrix['confuse_matrix']:
                    logging.error("vector no set")
                    return False
                dic_confuse_matrix['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_confuse_matrix['vector_path'])

                if 'image_path' not in dic_confuse_matrix['confuse_matrix']:
                    logging.error("image_path no set")
                    return False
                dic_confuse_matrix['image_path'] = '%s/%s' % (dic_data_dir['EVALUTION_DIR'], dic_confuse_matrix['image_path'])


            if 'roc' in self.dic_config['evalution']['logistic']['evaluate']:
                dic_roc = self.dic_config['evalution']['logistic']

                if 'predict_path' not in dic_roc['roc']:
                    logging.error("predict_path no set")
                    return False
                dic_roc['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_roc['predict_path'])

                if 'vector_path' not in dic_roc['roc']:
                    logging.error("vector no set")
                    return False
                dic_roc['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_roc['vector_path'])

                if 'image_path' not in dic_roc['roc']:
                    logging.error("image_path no set")
                    return False
                dic_roc['image_path'] = '%s/%s' % (dic_data_dir['EVALUTION_DIR'], dic_roc['image_path'])


            if 'f1' in self.dic_config['evalution']['logistic']['evaluate']:
                dic_f1 = self.dic_config['evalution']['logistic']

                if 'predict_path' not in dic_f1['roc']:
                    logging.error("predict_path no set")
                    return False
                dic_f1['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_f1['predict_path'])

                if 'vector_path' not in dic_f1['roc']:
                    logging.error("vector no set")
                    return False
                dic_f1['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_f1['vector_path'])


        if 'ai_xgboost' in self.dic_config['evalution']['model']:

            if 'confuse_matrix' in self.dic_config['evalution']['ai_xgboost']['evaluate']:
                dic_confuse_matrix = self.dic_config['evalution']['ai_xgboost']

                if 'predict_path' not in dic_confuse_matrix['confuse_matrix']:
                    logging.error("predict_path no set")
                    return False
                dic_confuse_matrix['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_confuse_matrix['predict_path'])

                if 'raw_data' not in dic_confuse_matrix['confuse_matrix']:
                    logging.error("raw_data no set")
                    return False
                dic_confuse_matrix['raw_data'] = '%s/%s' % (dic_data_dir['RAW_DIR'], dic_confuse_matrix['raw_data'])

                if 'vector_path' not in dic_confuse_matrix['confuse_matrix']:
                    logging.error("vector no set")
                    return False
                dic_confuse_matrix['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_confuse_matrix['vector_path'])

                if 'image_path' not in dic_confuse_matrix['confuse_matrix']:
                    logging.error("image_path no set")
                    return False
                dic_confuse_matrix['image_path'] = '%s/%s' % (dic_data_dir['EVALUTION_DIR'], dic_confuse_matrix['image_path'])


            if 'roc' in self.dic_config['evalution']['ai_xgboost']['evaluate']:
                dic_roc = self.dic_config['evalution']['logistic']

                if 'predict_path' not in dic_roc['roc']:
                    logging.error("predict_path no set")
                    return False
                dic_roc['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_roc['predict_path'])

                if 'vector_path' not in dic_roc['roc']:
                    logging.error("vector no set")
                    return False
                dic_roc['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_roc['vector_path'])

                if 'image_path' not in dic_roc['roc']:
                    logging.error("image_path no set")
                    return False
                dic_roc['image_path'] = '%s/%s' % (dic_data_dir['EVALUTION_DIR'], dic_roc['image_path'])

            if 'f1' in self.dic_config['evalution']['ai_xgboost']['evaluate']:
                dic_f1 = self.dic_config['evalution']['ai_xgboost']

                if 'predict_path' not in dic_f1['f1']:
                    logging.error("predict_path no set")
                    return False
                dic_f1['predict_path'] = '%s/%s' % (dic_data_dir['PREDICT_DIR'], dic_f1['predict_path'])

                if 'vector_path' not in dic_f1['f1']:
                    logging.error("vector no set")
                    return False
                dic_f1['vector_path'] = '%s/%s' % (dic_data_dir['PREPROCESS_DIR'], dic_f1['vector_path'])

            return True

    def check(self):
        if 'create_data' in self.dic_config['task']:
            rt = self.create_data()
            if not rt:
                logging.error('create_data config error')
                return {}

        if 'preprocess' in self.dic_config['task']:
            rt = self.preprocess()
            if not rt:
                logging.error('preprocess config error')
                return {}

        elif 'feature' in self.dic_config['task']:
            rt = self.feature()
            if not rt:
                logging.error('feature config error')
                return {}

        elif 'train' in self.dic_config['task']:
            rt = self.train()
            if not rt:
                logging.error('train config error')
                return {}

        elif 'predict' in self.dic_config['task']:
            if 'predict' not in self.dic_config:
                logging.error('predict config error')
                return {}
            rt = self.predict()
            if not rt:
                logging.error('predict config error')
                return {}

        elif 'evalution' in self.dic_config['task']:
            rt = self.evalution()
            if not rt:
                logging.error('evalution config error')
                return {}

        return self.dic_config

