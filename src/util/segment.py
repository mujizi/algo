# -*- coding: utf-8 -*-
# @Time    : 2018/10/15 上午10:53
# @Author  : Benqi

import fileinput
import jieba
import jieba.analyse

def jieba_init(setting):
    if 'JIEBA' not in setting:
        return

    if not setting.get('isJieba'):
        return

    dic_jieba = setting['JIEBA']
    if 'user_word' in dic_jieba:
        jieba.load_userdict(dic_jieba['user_word'])

    if 'stop_word' in dic_jieba:
        jieba.analyse.set_stop_words(dic_jieba['stop_word'])

        with open(dic_jieba['stop_word']) as f:
            stopwords = filter(lambda x: x, map(lambda x: x.strip().decode('utf-8'), f.readlines()))
        stopwords.extend([' ', '\t', '\n'])
        dic_jieba['stop_words'] = frozenset(stopwords)

    if 'tag' in dic_jieba:
        tag_file = dic_jieba['tag']
        dic_jieba['tag'] = {}
        for line in fileinput.input(tag_file):
            line = line.strip("\n").strip("\r")
            if not line:
                continue

            word = line.split('\t')
            word[1] = word[1].decode('utf8')
            dic_jieba['tag'][word[1]] = word[0]