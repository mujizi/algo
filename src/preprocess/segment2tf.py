# -*- coding: utf-8 -*-
# @Time    : 2018/10/15 下午5:05
# @Author  : Benqi
"""
segment file to translate into tf-idf vector space
"""
import os
import numpy as np
import pandas as pd

from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer

from base import Base
import util.tools as tools
import util.load_data as load_data

class Segment2tf(Base):

    """from batch to tfidf_vector
        data_form: batch： is used to deal with a batch
                   sample: is used to deal with a sample data

        tfidfspace.label: file's label
        tfidfspace.filenames: Direcory file's name
        tfidfspace.target_name: category name
        tfidfspace.vocabulary: vocabulary space
        tfidfspace.metrix: from batch text to batch tf_space matirx

    """

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_data(self):
        pass

    # dataset转化为tf——idf向量
    def create(self):
        if self.dic_config['data_form'] == 'batch':
            bunch = load_data.read_bunch(self.dic_config['bunch_path'])
            tfidfspace = Bunch(target_name=bunch.target_name,
                               label=bunch.label,
                               filenames=bunch.filenames,
                               metrix=[],
                               vocabulary={})

            if 'tfidfspace_train' in self.dic_config and self.dic_config['tfidfspace_train'] is not None:
                bunch_train = load_data.read_bunch(self.dic_config['tfidfspace_train'])
                tfidfspace.vocabulary = bunch_train.vocabulary
                vectorizer = TfidfVectorizer(sublinear_tf=True,
                                             max_df=0.5,
                                             vocabulary=bunch_train.vocabulary)

                tfidfspace.metrix = vectorizer.fit_transform(bunch.contents)
                load_data.write_bunch(self.dic_config['vector_path'], tfidfspace)
                self.logging.info('权重矩阵向量形状：{}'.format(tfidfspace.metrix.shape))

            else:
                vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
                tfidfspace.metrix = vectorizer.fit_transform(bunch.contents)
                tfidfspace.vocabulary = vectorizer.vocabulary_

                self.logging.info('权重矩阵向量形状：{}'.format(tfidfspace.metrix.shape))
                load_data.write_bunch(self.dic_config['vector_path'], tfidfspace)

                # 存入vocabulary词表到可视化文件
                with open(self.dic_config['visual_vector'], 'w') as f:
                    for i in tfidfspace.vocabulary.keys():
                        line = '%s,%s\n' % (i, tfidfspace.vocabulary[i])
                        f.write(line.encode('utf8'))

        elif self.dic_config['data_form'] == 'sample':
            bunch = load_data.read_bunch(self.dic_config['tfidfspace_train'])
            content = load_data.read_file(self.dic_config['pre_data'])

            tfidfspace = Bunch(metrix=[], vocabulary={})
            tfidfspace.vocabulary = bunch.vocabulary
            vectorizer = TfidfVectorizer(sublinear_tf=True,
                                         max_df=0.5,
                                         vocabulary=tfidfspace.vocabulary)

            tfidfspace.metrix = vectorizer.fit_transform([content])
            load_data._write_bunch_file(self.dic_config['vector_path'], tfidfspace)

