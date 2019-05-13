#!/bin/bash

## wangbenqi
ROOT_DIR='/Users/ouhon/PycharmProjects/ai_algo'
PYTHON='/usr/local/bin/python'


## yanzhiwu
#ROOT_DIR='/Users/yan/ai/ai_algo'
#PYTHON='/usr/bin/python'

DATA_DIR=$ROOT_DIR/data
SRC_DIR=$ROOT_DIR/src
LOG_DIR=$ROOT_DIR/log

mkdir -p $LOG_DIR

cd $ROOT_DIR/src
$PYTHON index.py --log $LOG_DIR/ai.log \
                 --config conf.config_face


