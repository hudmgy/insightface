import cv2
import os, sys
import numpy as np
import datetime
import glob
import os.path as osp
from skimage import transform as trans
from retinaface import RetinaFace
import ipdb


shape112x96 = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041] ], dtype=np.float32)
size112x96 = (112, 96)


shape112x112 = shape112x96
shape112x112[:,0] += 8.0
size112x112 = (112, 112)


as_shape128x128 = np.array([
    [40.3928, 40.2617],
    [87.3757, 40.0019],
    [64.0336, 66.9821],
    [44.7324, 94.4873],
    [83.6399, 94.2721] ], dtype=np.float32)
as_size128x128 = (128, 128)


as_shape112x112 = as_shape128x128 * 112 / 128
as_size112x112 = (112, 112)


def list_read(img_path):
    img_list = []
    for dirname,_,files in os.walk(img_path):
        for fi in files:
            if fi.endswith('.jpg'):
                img_list.append(osp.join(dirname, fi))
    return img_list


def detect(detector, scales, img, thresh, flip=False):
    h,w,c = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min((h,w))
    im_size_max = np.max((h,w))
    norm_img = np.zeros((im_size_max,im_size_max,c), dtype=img.dtype)
    norm_img[:h,:w] = img

    im_scale = float(target_size) / float(im_size_max)
    in_scales = [im_scale]
    faces, landmarks = detector.detect(norm_img, thresh, scales=in_scales, do_flip=flip)
    print(faces.shape, landmarks.shape)
    return faces, landmarks


def padding_crop(img, faces, landmarks, factor, std_size=None):
    crop_imgs = []
    h,w,_ = img.shape
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
            if std_size is not None:
                crop = cv2.resize(crop, std_size)
            crop_imgs.append(crop)
    return crop_imgs


def face_align(img, std_shape, image_size, bbox=None, landmark=None):
    M = None
    if landmark is not None:
        src = std_shape.copy()
        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
 
    if M is None:
        if bbox is None: #use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret 
    else: #do align using landmark
        warped = cv2.warpAffine(img, M, (image_size[1],image_size[0]), borderValue = 0.0)
        return warped


def align_crop(img, faces, landmarks):
    std_shape = as_shape112x112.copy()
    std_size = list(as_size112x112)

    margin_w, margin_h = 8, 8
    std_shape[:,0] += margin_w
    std_shape[:,1] += margin_h
    std_size[0] += 2 * margin_h
    std_size[1] += 2 * margin_w

    crop_imgs = []
    for i in range(faces.shape[0]):
        landmark5 = landmarks[i].astype(np.int)
        box = faces[i].astype(np.int)
        crop = face_align(img, std_shape, std_size, bbox=box, landmark=landmark5)
        crop_imgs.append(crop)
    return crop_imgs



def drawing(img, faces, landmarks):
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
    return img


def save_image(filename, img):
    if not osp.exists(osp.dirname(filename)):
        os.makedirs(osp.dirname(filename))
    cv2.imwrite(filename, img)



if __name__=='__main__':
    thresh = 0.8
    #scales = [1024, 1980]
    scales = [512, 990]
    #factor = 1.25
    #factor = 1.60
    factor = 1.50
    #std_size = (112, 112)
    std_size = (144, 144)
    
    img_path = sys.argv[1]
    gpuid = int(sys.argv[2])
    drawing = False 

    detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
    img_list = list_read(img_path)

    for ind, fi in enumerate(img_list):
        crop_file = fi.replace('raw', 'crop128x128')
        #crop_file = fi.replace('Data', 'crops')
        if osp.exists(crop_file): continue
        img = cv2.imread(fi)
        if img is None: continue

        faces, landmarks = detect(detector, scales, img, thresh)
        #crop_imgs = padding_crop(img, faces, landmarks, factor, std_size=std_size)
        crop_imgs = align_crop(img, faces, landmarks)
        print('found', len(crop_imgs), 'faces')
        for i in range(len(crop_imgs)):
            if i > 0:
                #crop_fi = osp.splitext(crop_file)[0]+'_%02d.jpg'%i
                pass
            else:
                crop_fi = crop_file
            save_image(crop_fi, crop_imgs[i])

        if drawing:
            img = drawing(img, faces, landmarks)
            drawing_file = crop_file.replace('crops', 'Drawing')
            save_image(drawing_file, img)
