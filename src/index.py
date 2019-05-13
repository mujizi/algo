# -*- coding:utf-8 -*-
# WEB 入口文件

from config import Config
import logging
from optparse import OptionParser

parser = OptionParser(usage="%prog [options]")
parser.add_option("-l", "--log", action="store", type="string", dest="log", help="log日志")
parser.add_option("-d", "--config", action="store", type="string", dest="config", help="config")
(options, args) = parser.parse_args()

# 动态倒入模块
import importlib
c = importlib.import_module(options.config)
CONFIG = c.CONFIG

g_config = {}

#初始化log
def init():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        filename=options.log,
                        filemode='w')

    dic_command = {}
    _config = Config()
    _config.init(dic_command, CONFIG)
    _config.set_data_dir()
    dic_config = _config.check()
    g_config.update(dic_config)

def create_data():
    if 'task' not in g_config['create_data'] or not g_config['create_data']['task']:
        return

    from create_data.create_data import Create_data
    Create_data(g_config['create_data']).run()


# 预处理task
def preprocess():
    if 'task' not in g_config['preprocess'] or not g_config['preprocess']['task']:
        return

    from preprocess.preprocess import Preprocess
    Preprocess(g_config['preprocess']).run()


def feature():
    if 'task' not in g_config['feature'] or not g_config['feature']['task']:
        return

    from feature.feature import Feature
    Feature(g_config['feature']).run()


def train():
    if 'model' not in g_config['train'] or not g_config['train']['model']:
        return

    from train.train import Trainer
    Trainer(g_config['train']).run()

def predict():
    if 'model' not in g_config['predict'] or not g_config['predict']['model']:
        return

    from predict.predict import Predict
    Predict(g_config['predict']).run()


def evalution():
    if 'model' not in g_config['evalution'] or not g_config['train']['model']:
        return

    from evalution.evalution import Evalution
    Evalution(g_config['evalution']).run()


def run():
    if 'task' not in g_config:
        return

    for task_name in g_config['task']:
        if task_name not in g_config:
            logging.warning('%s not exist' % task_name)
            continue
        eval(task_name)()


if __name__ == '__main__':
    init()
    run()

