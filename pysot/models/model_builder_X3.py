# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.utils.deploy_helper import print_properties


class ModelBuilder(object):
    def __init__(self, model):
        super(ModelBuilder, self).__init__()
        self.init_arch(model)

    def init_arch(self, model):
        self.inference = model['inference']

    def template(self, z):
        self.z = z

    def track(self, x):
        reg, cls1, cls2 = self.inference[0].forward([self.z, x])

        # print('reg feature map:')  # (1, 4, 11, 11)
        # print_properties(reg.properties)
        # print('cls1 feature map:')  # (121, 1, 1, 1)
        # print_properties(cls1.properties)
        # print('cls2 feature map:')  # (121, 1, 1, 1)
        # print_properties(cls2.properties)

        return {
            'loc': reg.buffer,
            'cls1': cls1.buffer,
            'cls2': cls2.buffer
        }
