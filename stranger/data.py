from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import random
import logging
import sys
import numbers
import math
import sklearn
import datetime
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
import ipdb

logger = logging.getLogger()


class FaceImageIter(io.DataIter):

    def __init__(self, batch_size, data_shape,
                 path_featu = None,
                 shuffle=False, aug_list=None, mean = None,
                 rand_mirror = False, cutoff = 0, color_jittering = 0,
                 data_name='data', label_name='softmax_label', **kwargs):
        super(FaceImageIter, self).__init__()

        assert path_featu
        logging.info('loading recordio %s...', path_featu)

        in_featu = np.load(osp.join(path_featu, 'in_featu.npy'))
        out_featu = np.load(osp.join(path_featu, 'out_featu.npy'))
        in_featu = in_featu[:,:data_shape[1]]
        out_featu = out_featu[:,:data_shape[1]]
        print('data shape:', in_featu.shape, out_featu.shape)
        self.featu = np.vstack((in_featu, out_featu))
        self.label = [1]*in_featu.shape[0] + [0]*out_featu.shape[0]
        self.imgidx = list(range(len(self.label)))
        self.seq = self.imgidx

        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.provide_label = [(label_name, (batch_size,))]
        #print(self.provide_label[0][1])
        self.cur = 0
        self.nbatch = 0
        self.is_init = False

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = 0
        if self.shuffle:
          random.shuffle(self.seq)

    def num_samples(self):
      return len(self.seq)

    def next_sample(self):
        if self.cur >= len(self.seq):
            raise StopIteration
        idx = self.seq[self.cur]
        self.cur += 1
        featu = self.featu[idx]
        label = self.label[idx]
        return label, featu

    def next(self):
        if not self.is_init:
          self.reset()
          self.is_init = True
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        self.nbatch+=1
        batch_size = self.batch_size
        _,dim = self.data_shape
        batch_data = nd.empty((batch_size, dim))
        if self.provide_label is not None:
          batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, featu = self.next_sample()
                assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                #print(datum.shape)
                batch_data[i][:] = nd.array(featu)
                batch_label[i] = label
                i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size - i)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 2:
            raise ValueError('data_shape should have length 2')
        if not data_shape[0] == 1:
            raise ValueError('This iterator expects inputs to have 1 channels.')
