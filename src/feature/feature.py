# -*- coding: utf-8 -*-
# @Author  : Benqi
# @Time    : 2018/10/25 上午11:26

# 预测入口类
class Feature():
    def __init__(self, dic_config={}):
        self.dic_config = dic_config

    def set_format(self, name):
        if name == 'hamming':
            import hamming
            self.feature = hamming.Hamming(self.dic_config[name])

        if name == 'jaccard':
            import jaccard
            self.feature = jaccard.Jaccard(self.dic_config[name])

        if name == 'normalization':
            import normalization
            self.feature = normalization.Normalization(self.dic_config[name])

        if name == 'standardization':
            import standardization
            self.feature = standardization.Standardization(self.dic_config[name])

        if name == 'cos_similarity':
            import cos_similarity
            self.feature = cos_similarity.Cos_similarity(self.dic_config[name])

        if name == 'tf_idf':
            import tf_idf
            self.feature = tf_idf.Tf_idf(self.dic_config[name])

    def load_data(self):
        self.feature.load_data()

    def feature_engineer(self):
        self.feature.feature_engineer()

    def dump(self):
        pass

    def run(self):
        for name in self.dic_config['task']:
            self.set_format(name)
            self.load_data()
            self.feature_engineer()
            self.dump()
