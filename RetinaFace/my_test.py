import cv2
import os, sys
import numpy as np
import datetime
import glob
import os.path as osp
import ipdb
from retinaface import RetinaFace

thresh = 0.8
#scales = [1024, 1980]
scales = [512, 990]

img_path = sys.argv[1]
gpuid = int(sys.argv[2])
drawing = False 
#factor = 1.25
factor = 1.60
#norm_size = 112
norm_size = 144


detector = RetinaFace('./model/R50', 0, gpuid, 'net3')

img_list = []
for dirname,_,files in os.walk(img_path):
    for fi in files:
        if fi.endswith('.jpg'):
            img_list.append(osp.join(dirname, fi))

for ind, fi in enumerate(img_list):
    crop_file = fi.replace('raw', 'crops')
    #crop_file = fi.replace('Data', 'crops')
    if osp.exists(crop_file):
        continue
    img = cv2.imread(fi)
    if img is None:
        continue
    h,w,c = img.shape
    print(ind, '/', len(img_list), img.shape)
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min((h,w))
    im_size_max = np.max((h,w))
    norm_img = np.zeros((im_size_max,im_size_max,c), dtype=img.dtype)
    norm_img[:h,:w] = img

    im_scale = float(target_size) / float(im_size_max)
    in_scales = [im_scale]
    flip = False
    faces, landmarks = detector.detect(norm_img, thresh, scales=in_scales, do_flip=flip)
    print(faces.shape, landmarks.shape)

    if faces is not None:
        print('find', faces.shape[0], 'faces')
        if drawing:
            for i in range(faces.shape[0]):
                box = faces[i].astype(np.int)
                color = (0,0,255)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                if landmarks is not None:
                    landmark5 = landmarks[i].astype(np.int)
                for l in range(landmark5.shape[0]):
                    color = (0,0,255)
                    if l==0 or l==3:
                        color = (0,255,0)
                    cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
            filename = fi.replace('Data', 'Drawing')
            if not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            cv2.imwrite(filename, img)
        else:
            for i in range(faces.shape[0]):
                box = faces[i].astype(np.int)
                wh = box[2] - box[0] + 1
                ht = box[3] - box[1] + 1
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                if wh <= 0 or ht <= 0:
                    continue
                sz = (wh + ht) / 4 * factor
                if sz > 20 and cy-sz>0 and cx-sz>0 and cy+sz<h and cx+sz<w:
                    sz, cx, cy = int(sz), int(cx), int(cy)
                    crop = img[cy-sz:cy+sz, cx-sz:cx+sz]
                    crop = cv2.resize(crop, (norm_size, norm_size))
                    if i > 0:
                        crop_fi = osp.splitext(crop_file)[0]+'_%02d.jpg'%i
                    else:
                        crop_fi = crop_file
                    if not osp.exists(osp.dirname(crop_fi)):
                        os.makedirs(osp.dirname(crop_fi))
                    cv2.imwrite(crop_fi, crop)
