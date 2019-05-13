# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 下午7:24
# @Author  : Benqi
import os
import re
from base import Base


class Txt2Word(Base):
    """
    将分词后的txt文件提取word，组成一个很大的list，然后存到一个txt中，这个txt文件的每一行是一个中文词汇
    """
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_data(self):
        def txt2list(path):
            file_list = os.listdir(path)
            new_list = []

            for i in file_list:
                full_path = path + '/' + i
                with open(full_path, 'r') as f:
                    txt_file = f.read()

                txt_list = re.split(r'\s+', txt_file)
                new_list.extend(txt_list)

            return new_list

        self.word_list = txt2list(self.dic_config['pre_data'])

    def create(self):
        with open(self.dic_config['word_path'], 'w') as f:
            for i in self.word_list:
                f.writelines(i + '\n')

