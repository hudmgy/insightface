from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import sklearn
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
import flops_counter
from logger import Logger


logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--dataset', default=default.dataset,
                        help='dataset config')
    parser.add_argument('--network', default=default.network,
                        help='network config')
    parser.add_argument('--loss', default=default.loss, help='loss config')
    parser.add_argument('--id', default='', help='training id')
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset, args.loss)
    parser.add_argument('--models-root', default=default.models_root,
                        help='root directory to save model.')
    parser.add_argument('--pretrained', default=default.pretrained,
                        help='pretrained model to load')
    parser.add_argument('--pretrained-epoch', type=int,
                        default=default.pretrained_epoch, help='pretrained epoch to load')
    parser.add_argument('--ckpt', type=int, default=default.ckpt,
                        help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
    parser.add_argument('--verbose', type=int, default=default.verbose,
                        help='do verification testing and model saving every verbose batches')
    parser.add_argument('--lr', type=float, default=default.lr,
                        help='start learning rate')
    parser.add_argument('--lr-steps', type=str,
                        default=default.lr_steps, help='steps of lr changing')
    parser.add_argument('--wd', type=float,
                        default=default.wd, help='weight decay')
    parser.add_argument('--mom', type=float,
                        default=default.mom, help='momentum')
    parser.add_argument('--frequent', type=int,
                        default=default.frequent, help='')
    parser.add_argument('--per-batch-size', type=int,
                        default=default.per_batch_size, help='batch size in each context')
    parser.add_argument('--kvstore', type=str,
                        default=default.kvstore, help='kvstore setting')
    args = parser.parse_args()
    return args


def get_symbol(args):
    embedding = eval(config.net_name).get_symbol()
    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label
    is_softmax = True

    if config.loss_name == 'softmax':  # softmax
        _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
                                     lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
        if config.fc7_no_bias:
            fc7 = mx.sym.FullyConnected(
                data=embedding, weight=_weight, no_bias=True, num_hidden=config.num_classes, name='fc7')
        else:
            _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
            fc7 = mx.sym.FullyConnected(
                data=embedding, weight=_weight, bias=_bias, num_hidden=config.num_classes, name='fc7')
    elif config.loss_name == 'margin_softmax':
        _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
                                     lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
        s = config.loss_s
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(
            embedding, mode='instance', name='fc1n')*s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight,
                                    no_bias=True, num_hidden=config.num_classes, name='fc7')
        if config.loss_m1 != 1.0 or config.loss_m2 != 0.0 or config.loss_m3 != 0.0:
            if config.loss_m1 == 1.0 and config.loss_m2 == 0.0:
                s_m = s*config.loss_m3
                gt_one_hot = mx.sym.one_hot(
                    gt_label, depth=config.num_classes, on_value=s_m, off_value=0.0)
                fc7 = fc7-gt_one_hot
            else:
                zy = mx.sym.pick(fc7, gt_label, axis=1)
                cos_t = zy/s
                t = mx.sym.arccos(cos_t)
                if config.loss_m1 != 1.0:
                    t = t*config.loss_m1
                if config.loss_m2 > 0.0:
                    t = t+config.loss_m2
                body = mx.sym.cos(t)
                if config.loss_m3 > 0.0:
                    body = body - config.loss_m3
                new_zy = body*s
                diff = new_zy - zy
                diff = mx.sym.expand_dims(diff, 1)
                gt_one_hot = mx.sym.one_hot(
                    gt_label, depth=config.num_classes, on_value=1.0, off_value=0.0)
                body = mx.sym.broadcast_mul(gt_one_hot, diff)
                fc7 = fc7+body
    elif config.loss_name == 'direct_regress':
        is_softmax = False
        _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
                                     lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
        s = config.loss_s
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(
            embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight,
                                    no_bias=True, num_hidden=config.num_classes, name='fc7')
    elif config.loss_name.find('triplet') >= 0:
        is_softmax = False
        nembedding = mx.symbol.L2Normalization(
            embedding, mode='instance', name='fc1n')
        anchor = mx.symbol.slice_axis(
            nembedding, axis=0, begin=0, end=args.per_batch_size//3)
        positive = mx.symbol.slice_axis(
            nembedding, axis=0, begin=args.per_batch_size//3, end=2*args.per_batch_size//3)
        negative = mx.symbol.slice_axis(
            nembedding, axis=0, begin=2*args.per_batch_size//3, end=args.per_batch_size)
        if config.loss_name == 'triplet':
            ap = anchor - positive
            an = anchor - negative
            ap = ap*ap
            an = an*an
            ap = mx.symbol.sum(ap, axis=1, keepdims=1)  # (T,1)
            an = mx.symbol.sum(an, axis=1, keepdims=1)  # (T,1)
            triplet_loss = mx.symbol.Activation(
                data=(ap-an+config.triplet_alpha), act_type='relu')
            triplet_loss = mx.symbol.mean(triplet_loss)
        else:
            ap = anchor*positive
            an = anchor*negative
            ap = mx.symbol.sum(ap, axis=1, keepdims=1)  # (T,1)
            an = mx.symbol.sum(an, axis=1, keepdims=1)  # (T,1)
            ap = mx.sym.arccos(ap)
            an = mx.sym.arccos(an)
            triplet_loss = mx.symbol.Activation(
                data=(ap-an+config.triplet_alpha), act_type='relu')
            triplet_loss = mx.symbol.mean(triplet_loss)
        triplet_loss = mx.symbol.MakeLoss(triplet_loss)
    out_list = [mx.symbol.BlockGrad(embedding)]

    if is_softmax:
        softmax = mx.symbol.SoftmaxOutput(
            data=fc7, label=gt_label, name='softmax', normalization='valid')
        out_list.append(softmax)
        if config.ce_loss:
            #ce_loss = mx.symbol.softmax_cross_entropy(data=fc7, label = gt_label, name='ce_loss')/args.per_batch_size
            body = mx.symbol.SoftmaxActivation(data=fc7)
            body = mx.symbol.log(body)
            _label = mx.sym.one_hot(
                gt_label, depth=config.num_classes, on_value=-1.0, off_value=0.0)
            body = body*_label
            ce_loss = mx.symbol.sum(body)/args.per_batch_size
            out_list.append(mx.symbol.BlockGrad(ce_loss))
    else:
        if config.loss_name == 'direct_regress':
            softmax = mx.symbol.SoftmaxOutput(
                data=fc7, label=gt_label, name='softmax', normalization='valid')
            out_list.append(softmax)
            _label = mx.sym.one_hot(
                gt_label, depth=config.num_classes, on_value=-1.0, off_value=1.0)
            body = fc7 * _label
            dr_loss = mx.symbol.sum(body)/args.per_batch_size
            out_list.append(mx.symbol.BlockGrad(dr_loss))
        else:
            out_list.append(mx.sym.BlockGrad(gt_label))
            out_list.append(triplet_loss)
    out = mx.symbol.Group(out_list)
    return out


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


    model = mx.mod.Module(
        context=ctx,
        symbol=sym,
    )


    print('loading', args.pretrained, args.pretrained_epoch)
    _, arg_params, aux_params = mx.model.load_checkpoint(
        args.pretrained, args.pretrained_epoch)
    sym = get_symbol(args)

    all_layers = model.symbol.get_internals()
    _sym = all_layers['fc1_output']
    _arg = {}
    for k in arg: 
        if not k.startswith('fc7'):
            _arg[k] = arg[k]
            mx.model.save_checkpoint(prefix, 0, _sym, _arg, aux)


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
