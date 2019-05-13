# -*- coding: utf-8 -*-
# @Author  : Benqi
# @Time    : 2018/10/25 上午11:26

# 预处理入口
class Preprocess():
    def __init__(self, dic_config={}):
        self.dic_config = dic_config

    def set_format(self, name):
        if name == 'raw2segment':
            import raw2segment
            self.file = raw2segment.Raw2segment(self.dic_config[name])

        if name == 'segment2tf':
            import segment2tf
            self.file = segment2tf.Segment2tf(self.dic_config[name])

        if name == 'h5py2bunch':
            import h5py2bunch
            self.file = h5py2bunch.H5py2bunch(self.dic_config[name])

        if name == 'encoder':
            import coder
            self.file = coder.Coder(self.dic_config[name])

        if name == 'csv2dat':
            import csv2dat
            self.file = csv2dat.Csv2dat(self.dic_config[name])

        if name == 'txt2bunch':
            import txt2bunch
            self.file = txt2bunch.Txt2bunch(self.dic_config[name])

        if name == 'segment2bunch':
            import segment2bunch
            self.file = segment2bunch.Segment2bunch(self.dic_config[name])

        if name == 'bunch2csv':
            import bunch2csv
            self.file = bunch2csv.Bunch2csv(self.dic_config[name])

        if name == 'package_dataset':
            import package_dataset
            self.file = package_dataset.Package_dataset(self.dic_config[name])

        if name == 'img_reshape':
            import img_reshape
            self.file = img_reshape.Img_reshape(self.dic_config[name])

        if name == 'img2bunch':
            import img2bunch
            self.file = img2bunch.Img2bunch(self.dic_config[name])

        if name == 'csv2bunch':
            import csv2bunch
            self.file = csv2bunch.Csv2bunch(self.dic_config[name])

        if name == 'matrix2sparse':
            import matrix2sparse
            self.file = matrix2sparse.Matrix2sparse(self.dic_config[name])

        if name == 'txt2tf':
            import txt2tf
            self.file = txt2tf.Txt2tf(self.dic_config[name])

        if name == 'data_extract':
            import data_extract
            self.file = data_extract.Data_extract(self.dic_config[name])

        if name == 'series2supervised':
            import series2supervised
            self.file = series2supervised.Series2supervised(self.dic_config[name])

        if name == 'txt2word':
            import txt2word
            self.file = txt2word.Txt2Word(self.dic_config[name])

        if name == 'word2vec':
            import word2vec
            self.file = word2vec.Word2Vec(self.dic_config[name])

    def load_data(self):
        self.file.load_data()

    def create(self):
        self.file.create()

    def run(self):
        for name in self.dic_config['task']:
            self.set_format(name)
            self.load_data()
            self.create()