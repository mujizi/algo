# -*- coding: utf-8 -*-
# @Time    : 2018/10/12 下午12:09
# @Author  : Benqi
"""
jieba to segment the raw txt file or batch file
"""
# 功能模块
import os
import jieba

# 自定义模块
from base import Base
import util.load_data as load_data

class Raw2segment(Base):

    """from batch to segment files
            raw_data: Chinese text files
            pre_data: file processed with jieba and stop words
            this is a directory file
    """

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_data(self):
        pass

    # 文件预处理,分词
    def create(self):
        if self.dic_config['data_form'] == 'batch':
            cate_list = os.listdir(self.dic_config['raw_data'])

            for my_dir in cate_list:
                self.logging.info(my_dir)
                class_path = str(self.dic_config['raw_data']) + "/" + str(my_dir) + "/"  # 拼出分类子目录的路径如：train_corpus/art/
                seg_dir = str(self.dic_config['pre_data']) + "/" + str(my_dir) + "/"
                self.logging.info(class_path)
                self.logging.info(seg_dir)

                if not os.path.exists(seg_dir):
                    os.makedirs(seg_dir)
                file_list = os.listdir(class_path)

                for file_path in file_list:
                    fullname = class_path + file_path
                    content = load_data.read_file(fullname)
                    stopwords_list = load_data.load_stopwords(self.dic_config['stop_word'])
                    self.logging.info(stopwords_list)

                    content_seg = ' '.join([i for i in jieba.cut(content) if i not in stopwords_list])
                    load_data.write_file(seg_dir + file_path, content_seg)
                    # logging.info("玩命分词")
                    # logging.info(content_seg)                         # 查看内容使用

        elif self.dic_config['data_form'] == 'sample':
            content = load_data.read_file(self.dic_config['raw_data'])
            stopwords_list = load_data.load_stopwords(self.dic_config['stop_word'])
            content_seg = ' '.join([i for i in jieba.cut(content) if i not in stopwords_list])

            load_data.write_file(self.dic_config['pre_data'], content_seg)



