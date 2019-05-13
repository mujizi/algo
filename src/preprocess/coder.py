# -*- coding: utf-8 -*-
# @Time    : 2018/11/6 下午5:40
# @Author  : Benqi
"""
Is used to change the file encode
"""
from base import Base

class Coder(Base):

    """encoder used to encode file
            params: raw_data
                    encode_path: after encode's file path

    """

    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_data(self, encode='utf-8'):
        with open(self.dic_config['raw_data'], 'rb') as f:
            try:
                self.file = f.read().decode(encode)
            except Exception as e:
                self.logging.exception(e)
                self.file = f.read()

    def create(self):
        with open(self.dic_config['encode_path'], 'wb') as f:
            f.write(self.file.encode('utf-8'))

        self.logging.info('————encode successful————')
        self.logging.info('file path: %s' % (self.dic_config['encode_path']))