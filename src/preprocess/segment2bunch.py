# -*- coding: utf-8 -*-
# @Time    : 2018/10/15 下午3:24
# @Author  : Benqi
"""
input: jieba segment file
output: bunch
"""
from src.preprocess.base import Base

# 功能模块
import os
import pickle
from sklearn.datasets.base import Bunch

# 自定义模块
import util.load_data as ld

class Segment2bunch(Base):

    """from batch to bunch
        pre_data: directory file

        dataset: bunch form

        bunch: Bunch(target_name=[], label=[], filenames=[], contents=[])
        this container is used to get label and X.

    """

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_data(self):
        catelist = os.listdir(self.dic_config['raw_data'])
        self.bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
        self.bunch.target_name.extend(catelist)

        for label in catelist:
            class_path = str(self.dic_config['raw_data']) + "/" + str(label) + "/"
            file_list = os.listdir(class_path)
            for file_path in file_list:
                fullname = str(class_path) + str(file_path)  # 拼出文件名全路径
                self.logging.info(fullname)
                self.bunch.label.append(label)
                self.bunch.filenames.append(fullname)
                self.bunch.contents.append(ld.read_file_raw(fullname))
                # logging.info(type(bunch.contents))           # bunch是一个是list
        self.logging.info(type(self.bunch.contents[1]))

    def create(self):
        if 'bunch_path' in self.dic_config:
            with open(self.dic_config['bunch_path'], 'wb') as f:
                pickle.dump(self.bunch, f)
            self.logging.info("dataset成功")

        if 'visual_path' in self.dic_config:
            with open(self.dic_config['visual_path'], 'w') as f:
                f.write('样本标签：{}{}{}'.format(self.bunch.target_name, '\n', '\n'))
                f.write('文章数:{}{}{}'.format(len(self.bunch.label), '\n', '\n'))
                f.write(self.bunch.contents[0].encode('utf-8'))





