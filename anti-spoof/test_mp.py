from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import time
import random
import glob
import numpy as np
import mxnet as mx
import argparse
from metric import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from data_iter import FaceImageIter
from logger import Logger
import ipdb


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--model', default='models/model, 2', help='path to load model.')
parser.add_argument('--data-dir', default='', help='training set directory')
parser.add_argument('--test-batch-size', type=int, default=1, help='batch size in each context')
args = parser.parse_args()

#image_size = [112, 112]
image_size = [32, 32]
data_shape = (3, image_size[0], image_size[1])


def softmax(x):
    exp = np.exp(x)
    partition = exp.sum(axis=1,keepdims=True)
    return exp / partition


def evaluate(predictions_score, test_loader, keyword):
    Thres = [0.01*i for i in range(0, 100, 1)]
    pass_num = [0]*len(Thres)
    tp_num = [0]*len(Thres)
    tn_num = [0]*len(Thres)
    pos_n = 0
    neg_n = 0
    pred_score = softmax(predictions_score)[:,1]
    for idx in range(pred_score.shape[0]):
        img_path, pid = test_loader.dataset[idx]
        score = pred_score[idx]
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
    print(keyword)
    for i,T in enumerate(Thres):
        fn = pos_n - tp_num[i]
        tn = tn_num[i]
        pn = pass_num[i]
        print('%.2f\t FAR: %d/%d=%f\t AR: %d/%d=%f'%(T, fn, pos_n, fn/(1e-10+pos_n), tn, neg_n, tn/(1e-10+neg_n)))


def forward(model, path_to_test):
    test_loader = FaceImageIter(
        batch_size=args.test_batch_size,
        data_shape=data_shape,
        path_imgrec=path_to_test,
        center_crop=True,
        mean=None,)

    start =  time.time()
    predictions_score = []
    for dat in test_loader:
        data = mx.io.DataBatch(data=dat.data)
        model.forward(data, is_train=False)
        ret = model.get_outputs()[0].asnumpy()
        #ret = softmax(ret)
        #predictions_score.append(ret[:,1])
        predictions_score.append(ret)
    end = time.time()
    print('Running time: {} Seconds'.format(end-start))
    predictions_score = np.vstack(predictions_score)
    return predictions_score, test_loader
 

def inference_main():
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
    mx.model.save_checkpoint(vec[0], 0, sym, arg_params, aux_params) 

    # data
    results = dict()
    file_list = glob.glob(os.path.join(args.data_dir, '*.lst'))
    for path_to_test in file_list:
        keyword = path_to_test
        predictions_score, test_loader = forward(model, path_to_test)
        evaluate(predictions_score, test_loader, keyword)
        results[keyword] = predictions_score
    np.save('results.npy', results)


def test_main():
    '''
    results = np.load('results.npy').item()
    res_array = []
    for k, v in results.items():
        test_loader = FaceImageIter(
            batch_size=args.test_batch_size,
            data_shape=data_shape,
            path_imgrec=k,
            center_crop=True,
            mean=None,)
        evaluate(v, test_loader, k)
        res_array.append(v)
    res_array_mean = np.mean(res_array[:5], 0)
    '''

    #p1 = np.load('test_log/r18-softmax-anti-001_115.npy')
    p1 = np.load('test_log/shuffse-softmax-anti-cutcenter_114.npy')
    p2 = np.load('test_log/shuffse-softmax-anti-00x1_96.npy')
    #res_array = np.vstack([p1, p2])
    #res_array_max = res_array.max(0)
    res_array_mean = np.mean([p1, p2], 0)

    ipdb.set_trace()
    path_to_test = os.path.join(args.data_dir, "test.lst")
    test_loader = FaceImageIter(
        batch_size=args.test_batch_size,
        data_shape=data_shape,
        path_imgrec=path_to_test,
        center_crop=True,
        mean=None,)
    evaluate(res_array_mean, test_loader, 'cascade')
       

if __name__=='__main__':
    #inference_main()
    test_main()
