from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import numpy as np
import symbol_utils
import sklearn
import ipdb


def fcnet(num_classes, filter_list, **kwargs):
    data = mx.symbol.Variable(name="data")

    data = mx.sym.Flatten(data=data)
    layer = data
    for ind, num_node in enumerate(filter_list):
        layer = mx.sym.FullyConnected(data=layer, num_hidden=num_node, name='pre_layer%d'%ind)
        layer = mx.sym.BatchNorm(data=layer, fix_gamma=True, eps=2e-5, momentum=0.9, name='layer%d'%ind)
        layer = mx.sym.LeakyReLU(data=layer, act_type='prelu', name='layer%d_relu'%ind)

    fc1 = mx.sym.FullyConnected(data=layer, num_hidden=num_classes, name='fc1')
    #fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=0.9, name='fc1')
    #fc1 = mx.sym.LeakyReLU(data=fc1, act_type='prelu', name='fc1_relu')

    return fc1


def get_symbol(num_classes, num_layers, **kwargs):
    if num_layers==0:
        filter_list = []
    elif num_layers==1:
        filter_list=[20]
    elif num_layers==2:
        filter_list=[20,20]
    elif num_layers==3:
        filter_list=[20,40,20]
    elif num_layers==5:
        filter_list=[20,40,80,160,320]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return fcnet(num_classes, filter_list, **kwargs)

