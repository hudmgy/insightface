from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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
from imgaug import augmenters as iaa

import glob
import os.path as osp
from collections import defaultdict
import ipdb

logger = logging.getLogger()


class FaceImageIter(io.DataIter):
    def __init__(
        self,
        batch_size,
        data_shape,
        path_imgrec=None,
        shuffle=False,
        balance=False,
        aug_list=None,
        rand_mirror=False,
        rand_crop=False,
        center_crop=False,
        rand_flip_crop=False,
        specified_crop=None,
        cutoff=0,
        fetch_size=None,
        color_jittering=0,
        buffer_en=False,
        data_name="data",
        label_name="softmax_label",
        **kwargs
    ):
        super(FaceImageIter, self).__init__()

        self.buffer_en = buffer_en
        self.data_list = path_imgrec
        self._check_before_run()
        dataset, num_data_pids, num_data_imgs = self._process_dir(self.data_list)

        if True:
            print("=> {} loaded".format(path_imgrec))
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  dataset    | {:5d} | {:8d}".format(num_data_pids, num_data_imgs))
            print("  ------------------------------")

        self.num_instances = 4
        self.dataset = dataset
        self.num_data_pids = num_data_pids
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(self.dataset):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        # get minimum of single id
        min_num_per_id = len(self.dataset)
        for pid, t in self.index_dic.items():
            if len(t) < min_num_per_id:
                min_num_per_id = len(t)
        self.min_num_per_id = min_num_per_id

        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.balance = balance
        self.image_size = "%d,%d" % (data_shape[1], data_shape[2])
        self.rand_mirror = rand_mirror
        self.center_crop = center_crop
        if rand_crop:
            assert center_crop==False, 'random crop and center crop can not work simulately.'
        if center_crop:
            assert rand_crop==False, 'random crop and center crop can not work simulately.'
        self.rand_crop = rand_crop
        self.rand_flip_crop = rand_flip_crop
        self.specified_crop = specified_crop
        self.cutoff = cutoff
        if fetch_size is not None:
            self.min_pat = fetch_size[0]
            self.max_pat = fetch_size[1]
        self.color_jittering = color_jittering
        self.CJA = mx.image.ColorJitterAug(0.125, 0.125, 0.125)
        self.HJA = mx.image.HueJitterAug(0.1)
        self.RGA = mx.image.RandomGrayAug(0.2)        

        self.provide_label = [(label_name, (batch_size,))]
        # print(self.provide_label[0][1])
        self.cur = 0
        self.nbatch = 0
        self.is_init = False
        self.imgidx = range(len(self.dataset))

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.data_list):
            raise RuntimeError("'{}' is not available".format(self.data_list))
        """
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.probe_dir):
            raise RuntimeError("'{}' is not available".format(self.probe_dir))
        """

    def _process_dir(self, list_path):
        with open(list_path, "r") as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.strip().split(',')
            pid = int(pid)  # no need to relabel
            if self.buffer_en:
                img = self.read_image(img_path)
                dataset.append((img, pid))
            else:
                dataset.append((img_path, pid))
            pid_container.add(pid)
        num_imgs = len(dataset)
        num_pids = len(pid_container)
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, (
                "See code comment for explanation " + list_path + " %d" % pid
            )
        return dataset, num_pids, num_imgs

    def reset(self):
        if self.balance:
            self.reset_balance()
        else:
            self.reset_all()

    def reset_all(self):
        """Resets the iterator to the beginning of the data."""
        print("call reset()")
        self.cur = 0
        self.seq = self.imgidx
        if self.shuffle:
            random.shuffle(self.seq)

    def reset_balance(self):
        self.cur = 0
        ret = []
        for pid, t in self.index_dic.items():
            t = np.random.choice(t, size=self.min_num_per_id, replace=False)
            #if len(t) > 2 * self.min_num_per_id:
            #    t = np.random.choice(t, size=2*self.min_num_per_id, replace=False)
            ret.extend(t)
        if self.shuffle:
            random.shuffle(ret)
        self.seq = ret
        print(len(self.seq), len(self.dataset))

    def reset_kv(self):
        """Resets the iterator to the beginning of the data."""
        print("call reset()")
        self.cur = 0
        indices = np.random.permutation(self.num_identities)
        self.seq = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            self.seq.extend(t)

    def color_aug(self, img):    
        img = self.CJA(img)
        #img = self.HJA(img)
        img = self.RGA(img)
        return img

    def color_jitter_aug(self, img):
        if self.color_jittering > 1:
            _rd = random.randint(0, 1)
            if _rd == 1:
                img = self.compress_aug(img)
        img = img.astype("float32", copy=False)
        img = self.color_aug(img)
        if img.max() > 255:
            img = img * 255 / img.max()
        return img

    def compress_aug(self, img):
        buf = BytesIO()
        img = Image.fromarray(img.asnumpy(), "RGB")
        #q = random.randint(2, 20)
        q = random.randint(20, 90)
        img.save(buf, format="JPEG", quality=q)
        buf = buf.getvalue()
        img = Image.open(BytesIO(buf))
        return nd.array(np.asarray(img, "float32"))

    def cutoff_patch(self, img, cutoff):
        _rd = random.randint(0, 1)
        if _rd == 1:
            centerh = random.randint(0, img.shape[0] - 1)
            centerw = random.randint(0, img.shape[1] - 1)
            half = self.cutoff // 2
            starth = max(0, centerh - half)
            endh = min(img.shape[0], centerh + half)
            startw = max(0, centerw - half)
            endw = min(img.shape[1], centerw + half)
            # print(starth, endh, startw, endw, _data.shape)
            img[starth:endh, startw:endw, :] = 128
        return img
      
    def fetch_patch(self, img, min_size, max_size):
        h,w,_ = img.shape
        min_size = min(min(w, h), min_size)
        max_size = max(max(w, h), max_size)
        size = random.randint(int(min_size), int(max_size))
        if size < min_size or size > max_size:
            return img

        half = size // 2
        centerh = random.randint(half,  h - half)
        centerw = random.randint(half, w - half)
        starth = centerh - half
        endh = centerh + half
        startw = centerw - half
        endw = centerw + half
        patch = img.copy()
        patch[:,:,:] = 128
        patch[starth:endh, startw:endw, :] = img[starth:endh, startw:endw, :]
        return patch

    def random_resize(self, img, probability = 0.5,  minRatio = 0.3):
        if random.uniform(0, 1) > probability:
            return img

        ratio = random.uniform(minRatio, 1.0)
        h = img.shape[0]
        w = img.shape[1]
        new_h = int(h*ratio)
        new_w = int(w*ratio)

        img = cv2.resize(img, (new_w,new_h))
        img = cv2.resize(img, (w, h))
        return img

    def random_cropping(self, img, target_shape, is_random = True):
        target_w, target_h = target_shape
        height, width, _ = img.shape

        if is_random:
            start_x = random.randint(0, width - target_w)
            start_y = random.randint(0, height - target_h)
        else:
            start_x = ( width - target_w ) // 2
            start_y = ( height - target_h ) // 2
        zeros = img[start_y:start_y+target_h,start_x:start_x+target_w,:]
        return zeros

    def random_flip_crop(self, img, target_shape):
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

        img,_ = mx.image.center_crop(img, (112, 112))
        image = augment_img.augment_image(img.asnumpy())
        image = self.random_resize(image)
        image = self.random_cropping(image, target_shape, is_random=True)
        return mx.nd.array(image)

    def specified_crop(self, img, target_shape, coords):
        start_x = coords[0]
        start_y = coords[1]
        target_w, target_h = target_shape
        image = img[start_y:start_y+target_h,start_x:start_x+target_w,:]
        return image
      
    def mirror_aug(self, img):
        _rd = random.randint(0, 1)
        if _rd == 1:
            img = mx.ndarray.flip(data=img, axis=1)
        return img  

    def num_samples(self):
        return len(self.dataset)

    def next_sample(self):
        """Helper function for reading in next sample."""
        while True:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            if self.dataset is not None:
                if self.buffer_en:
                    img, label = self.dataset[idx]
                else:
                    img_path, label = self.dataset[idx]
                    img = self.read_image(img_path)
                img = self.imdecode(img)
                return label, img
              
    def next(self):
        if not self.is_init:
            self.reset()
            self.is_init = True
        """Returns the next batch of data."""
        # print('in next', self.cur, self.labelcur)
        self.nbatch += 1
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        if self.provide_label is not None:
            batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, _data = self.next_sample()
                #if _data.shape[0] != self.data_shape[1]:
                #    _data = mx.image.resize_short(_data, self.data_shape[1])
                if self.color_jittering > 0:
                    _data = self.color_jitter_aug(_data)
                if self.cutoff > 0:
                    _data = self.cutoff_patch(_data, self.cutoff)
                if self.rand_mirror:
                    _data = self.mirror_aug(_data)
                if self.rand_crop:
                    _data,_ = mx.image.random_crop(_data, (w, h))
                if self.rand_flip_crop:
                    _data = self.random_flip_crop(_data, (w, h))
                if self.specified_crop is not None:
                    _data = self.specified_crop(_data, (w, h), self.specified_crop)
                if self.center_crop:
                    _data,_ = mx.image.center_crop(_data, (w, h))
                if hasattr(self, 'min_pat') and hasattr(self, 'max_pat'):
                    _data = self.fetch_patch(_data, self.min_pat, self.max_pat)
                self.save_image(_data)
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug("Invalid image, skipping:  %s", str(e))
                    continue
                for datum in data:
                    assert (
                        i < batch_size
                    ), "Batch size must be multiples of augmenter output length"
                    # print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i < batch_size:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size - i)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError("data_shape should have length 3, with dimensions CxHxW")
        if not data_shape[0] == 3:
            raise ValueError("This iterator expects inputs to have 3 channels.")

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError("Data shape is wrong")

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s)  # mx.ndarray
        return img

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(fname, "rb") as fin:
            img = fin.read()
        return img

    def save_image(self, img):
        import cv2
        fname = 'ximgx/' + str(self.cur) + '.jpg'
        ximg = img.copy()
        img[:,:,0] = ximg[:,:,2]
        img[:,:,2] = ximg[:,:,0]
        cv2.imwrite(fname, img.asnumpy())

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))


class FaceImageIterList(io.DataIter):
    def __init__(self, iter_list):
        assert len(iter_list) > 0
        self.provide_data = iter_list[0].provide_data
        self.provide_label = iter_list[0].provide_label
        self.iter_list = iter_list
        self.cur_iter = None

    def reset(self):
        self.cur_iter.reset()

    def next(self):
        self.cur_iter = random.choice(self.iter_list)
        while True:
            try:
                ret = self.cur_iter.next()
            except StopIteration:
                self.cur_iter.reset()
                continue
            return ret


if __name__ == "__main__":
    from config import config, default, generate_config

    generate_config("y2", "anti", "softmax")
    data_shape = (config.image_shape[2], config.image_shape[0], config.image_shape[1])
    #data_shape = (3, 112, 112)
    data_shape = (3, 48, 48)

    data_dir = config.dataset_path
    image_size = config.image_shape[0:2]
    assert len(image_size) == 2
    assert image_size[0] == image_size[1]
    print('image_size', image_size)
    #path_imgrec = os.path.join(data_dir, "train.lst")
    path_imgrec = os.path.join(data_dir, "test.lst")

    # data loader
    train_dataiter = FaceImageIter(
        batch_size=32,
        data_shape=data_shape,
        path_imgrec=path_imgrec,
        shuffle=True,
        balance=True,
        rand_flip_crop=True,
        buffer_en=True,
        rand_mirror=config.data_rand_mirror,
        cutoff=config.data_cutoff,
        #fetch_size=[32, 112],
        color_jittering=config.data_color,
        images_filter=config.data_images_filter,
    )
    '''
    val_dataiter = FaceImageIter(
        batch_size=args.test_batch_size,
        data_shape=data_shape,
        path_imgrec=path_test_imgrec,
    )
    '''

    while True:
        ipdb.set_trace()
        batchx = train_dataiter.next()
