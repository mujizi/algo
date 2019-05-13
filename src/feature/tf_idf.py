# -*- coding: utf-8 -*-
# @Time    : 2018/11/23 下午3:23
# @Author  : Benqi

from sklearn.feature_extraction.text import TfidfVectorizer
import util.load_data as load_data

from base import Base

class Tf_idf(Base):
    def __init__(self, dic_config={}):
        Base.__init__(self, dic_config)

    def load_data(self):
        self.content = load_data.read_file(self.dic_config['data_path'])
        self.txt_list = self.content.split(' ')

    def feature_engineer(self):
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([self.content])

        matrix = matrix.toarray()


        # logging.info(type(matrix))
        self.logging.info(matrix.toarray())
        # logging.info(matrix)

        # metrix = metrix.toarray()
        #
        # metrix = np.squeeze(metrix).tolist()
        #
        # result = heapq.nlargest(self.dic_config['topk'], metrix)


    def dump(self):
        pass