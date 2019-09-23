from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nn_model
import os
import os.path as osp
import sys
import math
import random
import logging
import pickle
import numpy as np
import sklearn
from data import FaceImageIter
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
import ipdb
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))


logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None


class AccMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(AccMetric, self).__init__(
            'acc', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        self.count += 1
        label = labels[0].asnumpy()
        pred_label = preds[0].asnumpy()
        pred_label = np.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.astype('int32').flatten()
        label = label.astype('int32').flatten()
        assert label.shape == pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)


class LossValueMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(LossValueMetric, self).__init__(
            'lossvalue', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []

    def update(self, labels, preds):
        loss = preds[-1].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1.0
        gt_label = preds[-2].asnumpy()
        # print(gt_label)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--data-dir', default='',
                        help='training set directory')
    parser.add_argument('--prefix', default='model/model',
                        help='directory to save model.')
    parser.add_argument('--pretrained', default='',
                        help='pretrained model to load')
    parser.add_argument('--ckpt', type=int, default=1,
                        help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
    parser.add_argument('--loss-type', type=int, default=4, help='loss type')
    parser.add_argument('--verbose', type=int, default=2000,
                        help='do verification testing and model saving every verbose batches')
    parser.add_argument('--max-steps', type=int, default=0,
                        help='max training batches')
    parser.add_argument('--end-epoch', type=int,
                        default=900000, help='training epoch size.')
    parser.add_argument('--network', default='n2', help='specify network')
    parser.add_argument('--image-size', default='',
                        help='specify input image height and width')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='start learning rate')
    parser.add_argument('--lr-steps', type=str, default='40000,80000,120000',
                        help='steps of lr changing')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--per-batch-size', type=int,
                        default=128, help='batch size in each context')
    parser.add_argument('--rand-mirror', type=int, default=1,
                        help='if do random mirror in training')
    parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
    parser.add_argument('--color', type=int, default=0,
                        help='color jittering aug')
    parser.add_argument('--ce-loss', default=False,
                        action='store_true', help='if output ce loss')
    args = parser.parse_args()
    return args


def get_symbol(args, arg_params, aux_params):
    # data_shape = (args.image_channel, args.image_h, args.image_w)
    # image_shape = ",".join([str(x) for x in data_shape])

    fc1 = nn_model.get_symbol(num_classes=2, num_layers=5)
    label = mx.symbol.Variable('softmax_label')
    softmax = mx.symbol.SoftmaxOutput(data=fc1, label=label, name='softmax', normalization='valid', use_ignore=True, ignore_label=9999)
    # outs = [softmax]
    # outs.append(mx.sym.BlockGrad(fc1))
    # out = mx.symbol.Group(outs)

    return (softmax, arg_params, aux_params)


def train_net(args):
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
    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    #args.num_layers = int(args.network[1:])
    #print('num_layers', args.num_layers)
    if args.per_batch_size == 0:
        args.per_batch_size = 128
    args.batch_size = args.per_batch_size*args.ctx_num
    #args.rescale_threshold = 0
    #args.image_channel = 1

    path_data = os.path.join(args.data_dir, "train")

    print('Called with argument:', args)
    #data_shape = (args.image_channel, 7)
    mean = None

    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    if len(args.pretrained) == 0:
        arg_params = None
        aux_params = None
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    else:
        vec = args.pretrained.split(',')
        print('loading', vec)
        _, arg_params, aux_params = mx.model.load_checkpoint(
            vec[0], int(vec[1]))
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)

    model = mx.mod.Module(
        context=ctx,
        symbol=sym,
    )

    train_dataiter = data_loader(path_data, batch_size=args.batch_size, shuffle=True)
    val_dataiter = data_loader(path_data, batch_size=64, shuffle=False)

    metric = mx.metric.CompositeEvalMetric([AccMetric()])
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2)
    _rescale = 1.0/args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr,
                        momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    #opt = optimizer.Nadam(learning_rate=base_lr, wd=base_wd, rescale_grad=_rescale)
    som = 20
    _cb = mx.callback.Speedometer(args.batch_size, som)
    lr_steps = [int(x) for x in args.lr_steps.split(',')]

    global_step = [0]
    save_step = [0]
    def _batch_callback(param):
        _cb(param)
        global_step[0] += 1
        mbatch = global_step[0]
        for _lr in lr_steps:
            if mbatch == _lr:
                opt.lr *= 0.1
                print('lr change to', opt.lr)
                break
        if mbatch % 1000 == 0:
            print('lr-batch-epoch:', opt.lr, param.nbatch, param.epoch)
        if (mbatch >= 0 and mbatch % args.verbose == 0) or mbatch == lr_steps[-1]:
            save_step[0] += 1
            msave = save_step[0]
            arg, aux = model.get_params()
            all_layers = model.symbol.get_internals()
            _sym = all_layers['fc1_output']
            mx.model.save_checkpoint(args.prefix, msave, _sym, arg, aux)

    epoch_cb = None
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    print('start fitting')

    model.fit(train_dataiter,
              begin_epoch=begin_epoch,
              num_epoch=end_epoch,
              eval_data=val_dataiter,
              eval_metric=metric,
              kvstore='device',
              optimizer=opt,
              #optimizer_params   = optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              batch_end_callback=_batch_callback,
              epoch_end_callback=epoch_cb)


def main():
    # time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)


def data_loader(path_featu, batch_size=256, featu_dim=20, shuffle=False):
    in_featu = np.load(osp.join(path_featu, 'in_featu.npy'))
    out_featu = np.load(osp.join(path_featu, 'out_featu.npy'))
    in_featu = in_featu[:,:featu_dim]
    out_featu = out_featu[:,:featu_dim]
    print('data shape:', in_featu.shape, out_featu.shape)
    featu = np.vstack((in_featu, out_featu))
    label = [0]*in_featu.shape[0] + [1]*out_featu.shape[0]

    training_num = int(len(label)*0.8)
    random.seed(a=0)
    rid = random.sample(range(len(label)), len(label))
    if shuffle:
        featu = featu[rid[:training_num]]
        label = [label[r] for r in rid[:training_num]]
        print('training:', featu.shape)
    else:
        featu = featu[rid[training_num:]]
        label = [label[r] for r in rid[training_num:]]
        print('testing:', featu.shape)

    label = np.array(label)
    train_iter = mx.io.NDArrayIter(featu, label, batch_size, shuffle=shuffle)
    return train_iter


if __name__ == '__main__':
    main()
