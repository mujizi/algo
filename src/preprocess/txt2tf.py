# -*- coding: utf-8 -*-
# @Time    : 2018/12/10 下午8:36
# @Author  : Benqi

import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import Bunch

from base import Base
import util.load_data as load_data

class Txt2tf(Base):
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_data(self):
        pass

    # 文件预处理,分词
    def create(self):
        if self.dic_config['data_form'] == 'batch':
            cate_list = os.listdir(self.dic_config['raw_data'])
            self.bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
            self.bunch.target_name.extend(cate_list)

            for my_dir in cate_list:
                self.logging.info(my_dir)
                class_path = str(self.dic_config['raw_data']) + "/" + str(my_dir) + "/"  # 拼出分类子目录的路径如：train_corpus/art/
                self.logging.info(class_path)

                file_list = os.listdir(class_path)

                for file_path in file_list:
                    fullname = class_path + file_path
                    content = load_data.read_file(fullname)
                    stopwords_list = load_data.load_stopwords(self.dic_config['stop_word'])

                    content_seg = ' '.join([i for i in jieba.cut(content) if i not in stopwords_list])
                    # load_data.write_file(seg_dir + file_path, content_seg)
                    # logging.info("玩命分词")
                    # logging.info(content_seg)                         # 查看内容使用

                    self.bunch.label.append(my_dir)
                    self.bunch.filenames.append(fullname)
                    self.bunch.contents.append(content_seg)

            tfidfspace = Bunch(target_name=self.bunch.target_name,
                               label=self.bunch.label,
                               filenames=self.bunch.filenames,
                               metrix=[],
                               vocabulary={})

            if 'tfidfspace_train' in self.dic_config and self.dic_config['tfidfspace_train'] is not None:
                bunch_train = load_data.read_bunch(self.dic_config['tfidfspace_train'])
                tfidfspace.vocabulary = bunch_train.vocabulary
                vectorizer = TfidfVectorizer(sublinear_tf=True,
                                             max_df=0.5,
                                             vocabulary=bunch_train.vocabulary)

                tfidfspace.metrix = vectorizer.fit_transform(self.bunch.contents)
                load_data.write_bunch(self.dic_config['vector_path'], tfidfspace)
                self.logging.info('权重矩阵向量形状：{}'.format(tfidfspace.metrix.shape))

            else:
                vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
                tfidfspace.metrix = vectorizer.fit_transform(self.bunch.contents)
                tfidfspace.vocabulary = vectorizer.vocabulary_

                self.logging.info('权重矩阵向量形状：{}'.format(tfidfspace.metrix.shape))
                load_data.write_bunch(self.dic_config['vector_path'], tfidfspace)

                # 存入vocabulary词表到可视化文件
                with open(self.dic_config['visual_vector'], 'w') as f:
                    for i in tfidfspace.vocabulary.keys():
                        line = '%s,%s\n' % (i, tfidfspace.vocabulary[i])
                        f.write(line.encode('utf8'))


        elif self.dic_config['data_form'] == 'sample':
            content = load_data.read_file(self.dic_config['raw_data'])
            stopwords_list = load_data.load_stopwords(self.dic_config['stop_word'])
            content_seg = ' '.join([i for i in jieba.cut(content) if i not in stopwords_list])
            bunch = load_data.read_bunch(self.dic_config['tfidfspace_train'])
            tfidfspace = Bunch(metrix=[], vocabulary={})
            tfidfspace.vocabulary = bunch.vocabulary
            vectorizer = TfidfVectorizer(sublinear_tf=True,
                                         max_df=0.5,
                                         vocabulary=tfidfspace.vocabulary)

            tfidfspace.metrix = vectorizer.fit_transform([content_seg])
            load_data._write_bunch_file(self.dic_config['vector_path'], tfidfspace)


