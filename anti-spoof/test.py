from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import time
import random
import logging
#import sklearn
import pickle
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
from config import config, default, generate_config
from metric import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbol'))
import shufflenetv2
import fdensenet
import fmnasnet
import fmobilenet
import fmobilefacenet
import fresnet
import verification
from data_iter import FaceImageIter
import flops_counter
from logger import Logger
import ipdb


def softmax(x):
    exp = np.exp(x)
    partition = exp.sum(axis=1,keepdims=True)
    return exp / partition


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--model', default='models/model, 2', help='path to load model.')
parser.add_argument('--data-dir', default='', help='training set directory')
parser.add_argument('--test-batch-size', type=int, default=1, help='batch size in each context')
args = parser.parse_args()

#image_size = [112, 112]
image_size = [32, 32]
data_shape = (3, image_size[0], image_size[1])


ctx = []
cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
if len(cvd) > 0:
    for i in range(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
if len(ctx) == 0:
    ctx = [mx.cpu()]
    print('use cpu')
else:
    print('gpu num:', len(ctx))


# model
vec = args.model.split(',')
print('loading', vec)
sym, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
all_layers = sym.get_internals()
sym = all_layers['fc7'+'_output']
model = mx.mod.Module(context=ctx, symbol=sym)
model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
model.set_params(arg_params, aux_params)


# data loader
path_test_imgrec = os.path.join(args.data_dir, "test.lst")
test_loader = FaceImageIter(
    batch_size=args.test_batch_size,
    data_shape=data_shape,
    path_imgrec=path_test_imgrec,
    center_crop=True,
    mean=None,)

# 

start =  time.time()
predictions = []
predictions_score = []
for dat in test_loader:
    #label = dat.label[0].asnumpy()
    data = mx.io.DataBatch(data=dat.data)
    model.forward(data, is_train=False)
    ret = model.get_outputs()[0].asnumpy()
    ret = softmax(ret)
    preds = np.argmax(ret, 1)
    predictions.append(preds)
    predictions_score.append(ret[:,1])
end = time.time()
print('Running time: {} Seconds'.format(end-start))

Thres = [0.01*i for i in range(0, 100, 1)]
pass_num = [0]*len(Thres)
tp_num = [0]*len(Thres)
tn_num = [0]*len(Thres)
pos_n = 0
neg_n = 0
predictions = np.hstack(predictions)
predictions_score = np.hstack(predictions_score)
for idx in range(predictions_score.shape[0]):
    img_path, pid = test_loader.dataset[idx]
    score = predictions_score[idx]
    if pid==1:
        pos_n += 1
    else:
        neg_n += 1
    for i, T in enumerate(Thres):
        if score >= T:
            if pid==1:
                tp_num[i] += 1
        else:
            if pid==0:
                tn_num[i] += 1
for i,T in enumerate(Thres):
    fn = pos_n - tp_num[i]
    tn = tn_num[i]
    pn = pass_num[i]
    print('%.2f\t FAR: %d/%d=%f\t AR: %d/%d=%f'%(T, fn, pos_n, fn/(1e-10+pos_n), tn, neg_n, tn/(1e-10+neg_n)))


'''
with open(os.path.join(args.data_dir, 'pred.csv'), 'w') as fp:
    for idx, (img_path, pid) in enumerate(test_loader.dataset):
        fp.write('%s,%d,%d,%.3f\n'%(img_path, pid, predictions[idx], predictions_score[idx]))
print('Acc(%d/%d) = %.4f\nTime = %.4f'%(all_corrects, all_test, acc, end-start))
'''
mx.model.save_checkpoint(vec[0], 0, sym, arg_params, aux_params) 
