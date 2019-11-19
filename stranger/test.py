import argparse
import os,sys
import numpy as np
import datetime
from train import get_symbol, data_loader
import mxnet as mx
from mxnet import ndarray as nd
import ipdb


def softmax(x):
    exp = np.exp(x)
    partition = exp.sum(axis=1,keepdims=True)
    return exp / partition


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--model', default='model/20d/model,2', help='path to load model.')
parser.add_argument('--data-dir', default='',
                        help='training set directory')
args = parser.parse_args()

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


vec = args.model.split(',')
print('loading', vec)
sym, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
all_layers = sym.get_internals()
sym = all_layers['fc1'+'_output']
model = mx.mod.Module(context=ctx, symbol=sym)
model.bind(data_shapes=[('data', (1, 20))])
model.set_params(arg_params, aux_params)

path_data = os.path.join(args.data_dir, "train")
test_loader = data_loader(path_data, batch_size=64, shuffle=False)
acc_num = 0
number = 0
for dat in test_loader:
    label = dat.label[0].asnumpy()
    data = mx.io.DataBatch(data=dat.data)
    model.forward(data, is_train=False)
    ret = model.get_outputs()[0].asnumpy()
    ret = softmax(ret)
    pred = np.argmax(ret, 1)
    for i in range(ret.shape[0]):
        if ret[i,1] > 0.7:
            pred[i] = 1
            if pred[i]==label[i]:
                acc_num += 1
            number += 1
        else:
            pred[i] = 0
    #acc_num += sum(pred==label)
    #number += label.shape[0]

print('%d / %d = %f'%(acc_num, number, 1.0*acc_num/number))
