# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 下午5:13
# @Author  : Benqi
import sys
sys.path.append('../')

import util.load_data as load_data
import util.tools as tools

class Bayes():
    """bunch.metrix: like list, every list element is a sample data
       predicted: 1d numpy.array,like a list

    """
    def load_data(self, data_path):
        bunch = load_data.read_bunch(data_path)
        metrix = bunch.metrix
        return metrix

    def load_model(self, model_path):
        load = tools.Access_model()
        self.model = load.load_model(model_path)

    def predict(self, data_path):
        metrix = self.load_data(data_path)
        result = self.model.predict(metrix)

        return result

if __name__ == '__main__':
    bayes = Bayes()
    bayes.load_model('/Users/ouhon/PycharmProjects/ai_algo/data/bayes/train/bayes.m')
    result = bayes.predict('/Users/ouhon/PycharmProjects/ai_algo/data/bayes/preprocess/tfidfspace_test.dat')
    print(result)