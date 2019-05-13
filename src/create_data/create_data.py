# -*- coding: utf-8 -*-
# @Time    : 2018/12/3 上午10:49
# @Author  : Benqi

class Create_data():
    def __init__(self, dic_config={}):
        self.dic_config = dic_config

    def set_task(self, task_name):
        if task_name == 'captcha':
            import ai_captcha
            self.model = ai_captcha.AICaptcha(self.dic_config[task_name])

    def create(self):
        self.model.generate_data()

    def run(self):
        for task in self.dic_config['task']:
            self.set_task(task)
            self.create()