import argparse
import os,sys
import numpy as np
import datetime
from train import get_symbol, data_loader
import mxnet as mx
from mxnet import ndarray as nd
from collections import namedtuple
import ipdb


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--model', default='model/mx-model/model,0', help='path to load model.')
parser.add_argument('--prefix', default='model/mx-model/model',
                        help='directory to save model.')
parser.add_argument('--data-dir', default='',
                        help='training set directory')
args = parser.parse_args()


ctx = []
'''
cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
if len(cvd) > 0:
    for i in range(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
'''
if len(ctx) == 0:
    ctx = [mx.cpu()]
    print('use cpu')
else:
    print('gpu num:', len(ctx))



def softmax(x):
    exp = np.exp(x)
    partition = exp.sum(axis=1,keepdims=True)
    return exp / partition

def fcnet_with_featu(num_classes, filter_list=None):
    data = mx.symbol.Variable(name="data")
    data = mx.sym.Flatten(data=data)
    layer = data
    for ind, num_node in enumerate(filter_list):
        layer = mx.sym.FullyConnected(data=layer, num_hidden=num_node, bias=None, name='layer%d'%ind)
        layer = mx.sym.relu(data=layer, name='layer%d_relu'%ind)
    fc1 = mx.sym.FullyConnected(data=layer, num_hidden=num_classes, name='fc1')
    return fc1

def get_symbol(sim_dim=0):
    fc1 = fcnet_with_featu(2, filter_list=[256+sim_dim, 32+sim_dim, 16])
    #label = mx.symbol.Variable('softmax_label')
    #softmax = mx.symbol.SoftmaxOutput(data=fc1, label=label, name='softmax', normalization='valid', use_ignore=True, ignore_label=9999)
    #return softmax
    return fc1


sim_dim = 20
shapes = (1, 532)

'''
import torch
th_model_file = 'model/cfg_train_150k_20d_with_featu/checkpoint_ep24.pth.tar'
checkpoint = torch.load(th_model_file)
state_dict = checkpoint['state_dict']

flow = get_symbol(sim_dim=sim_dim)
name_mxnet = flow.list_arguments()
print(type(name_mxnet))
for key in name_mxnet:
    print(key)

#convert
#model = flow.simple_bind(ctx=ctx, data=shapes)
model = mx.mod.Module(context=ctx, symbol=flow)
model.bind(data_shapes=[('data', shapes)])
model.init_params(initializer=mx.initializer.Zero())
all_layers = model.symbol.get_internals()
sym = all_layers['fc1_output']
arg, aux = model.get_params()
for key in state_dict:
    weights = state_dict[key].data.cpu()
    if key=='layers.0.gc.weight':
        arg['layer0_weight'][:256,:512]=weights.permute(1,0)
        for i in range(sim_dim):
            arg['layer0_weight'][-i-1,-i-1] = 1
    elif key=='layers.1.gc.weight':
        arg['layer1_weight'][:32,:256]=weights.permute(1,0)
        for i in range(sim_dim):
            arg['layer1_weight'][-i-1,-i-1] = 1
    elif key=='layers_po.0.gc.weight':
        arg['layer2_weight'][:]=weights.permute(1,0)
    elif key=='classifier.weight':
        arg['fc1_weight'][:]=weights
    elif key=='classifier.bias':
        arg['fc1_bias'][:]=weights
model.set_params(arg, aux)
mx.model.save_checkpoint(args.prefix, 0, sym, arg, aux)
'''

# load
sym, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, 0)
all_layers = sym.get_internals()
sym = all_layers['fc1'+'_output']
model = mx.mod.Module(context=ctx, symbol=sym)
model.bind(data_shapes=[('data', shapes)])
model.set_params(arg_params, aux_params)

xdat = np.load('xdat.npy')
Batch = namedtuple('Batch', ['data'])
data = mx.sym.Variable('data')
xdat1 = mx.nd.array(xdat).astype(np.float32)
data1 = [xdat1]
#data1 = [mx.nd.zeros((1, 532))]
model.forward(Batch(data1), is_train=False)
# test
#model.forward(data, is_train=False)
ret = model.get_outputs()[0].asnumpy()
ipdb.set_trace()
print(ret)
ret = softmax(ret)
pred = np.argmax(ret, 1)
print(pred)
mx.model.save_checkpoint('model/mx-model/model99', 0, sym, arg_params, aux_params)
