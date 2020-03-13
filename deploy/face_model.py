from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_image
import face_preprocess


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, args):
    self.args = args
    ctx = mx.gpu(args.gpu)
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    self.ga_model = None
    if len(args.model)>0:
      self.model = get_model(ctx, image_size, args.model, 'fc1')
    if len(args.ga_model)>0:
      self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')

    self.threshold = args.threshold
    self.det_minsize = 50
    self.det_threshold = [0.6,0.7,0.8]
    #self.det_factor = 0.9
    self.image_size = image_size
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    if args.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
    else:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector


  def get_input(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    #bbox = bbox[0,0:4]
    #bbx selection
    bbw = bbox[:, 2] - bbox[:, 0] + 1
    bbh = bbox[:, 3] - bbox[:, 1] + 1
    area = bbw * bbh
    ind = np.where(area == np.max(area))
    bbox = bbox[ind[0][0],0:4]
    ###
    #points = points[0,:].reshape((2,5)).T
    points = points[ind[0][0],:].reshape((2,5)).T
    #print(bbox)
    #print(points)
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    return aligned

  def get_input_center_box(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None

    ##bbox = bbox[0,0:4]
    ##bbx selection

    #bbw = bbox[:, 2] - bbox[:, 0] + 1
    #bbh = bbox[:, 3] - bbox[:, 1] + 1
    #area = bbw * bbh
    #ind = np.where(area == np.max(area))
    #bbox = bbox[ind[0][0],0:4]
    
    nrof_faces = bbox.shape[0]
    det = bbox[:,0:4]
    img_size = np.asarray(face_img.shape)[0:2]

    if nrof_faces>1:
        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
        img_center = img_size / 2
        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0]  ])
        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
        det = det[index,:]
        points = points[index,:]
    det = np.squeeze(det)
    bbox = det
    #bb = np.zeros(4, dtype=np.int32)
    #bb[0] = np.maximum(det[0]-args.margin/2, 0)
    #bb[1] = np.maximum(det[1]-args.margin/2, 0)
    #bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
    #bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
    
    #cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    #scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
    #nrof_successfully_aligned += 1
    #misc.imsave(output_filename, scaled)
    #text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
    ###
    #points = points[0,:].reshape((2,5)).T
    points = points.reshape((2,5)).T
    #print(bbox)
    #print(points)
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    #nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    #aligned = np.transpose(nimg, (2,0,1))
    return nimg #aligned

  def get_input_center_box_return_bbx(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return None,None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None,None

    ##bbox = bbox[0,0:4]
    ##bbx selection

    #bbw = bbox[:, 2] - bbox[:, 0] + 1
    #bbh = bbox[:, 3] - bbox[:, 1] + 1
    #area = bbw * bbh
    #ind = np.where(area == np.max(area))
    #bbox = bbox[ind[0][0],0:4]
    
    nrof_faces = bbox.shape[0]
    det = bbox[:,0:4]
    img_size = np.asarray(face_img.shape)[0:2]

    if nrof_faces>1:
        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
        img_center = img_size / 2
        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0]  ])
        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
        det = det[index,:]
        points = points[index,:]
    det = np.squeeze(det)
    bbox = det
    #bb = np.zeros(4, dtype=np.int32)
    #bb[0] = np.maximum(det[0]-args.margin/2, 0)
    #bb[1] = np.maximum(det[1]-args.margin/2, 0)
    #bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
    #bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
    
    #cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    #scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
    #nrof_successfully_aligned += 1
    #misc.imsave(output_filename, scaled)
    #text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
    ###
    #points = points[0,:].reshape((2,5)).T
    points = points.reshape((2,5)).T
    #print(bbox)
    #print(points)
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    return aligned, bbox #aligned


 #get the largest bbox  
  def get_input_out_128(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    #bbox = bbox[0,0:4]
    #bbx selection
    bbw = bbox[:, 2] - bbox[:, 0] + 1
    bbh = bbox[:, 3] - bbox[:, 1] + 1
    area = bbw * bbh
    ind = np.where(area == np.max(area))
    bbox = bbox[ind[0][0],0:4]
    ###
    #points = points[0,:].reshape((2,5)).T
    points = points[ind[0][0],:].reshape((2,5)).T
    #print(bbox)
    #print(points)
    nimg = face_preprocess.preprocess_out_128(face_img, bbox, points, image_size='112,112')
    #nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    #aligned = np.transpose(nimg, (2,0,1))
    return nimg #aligned

  def get_input_out_128_all_bbx(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return [] 
    bboxs, points = ret
    if bboxs.shape[0]==0:
      return [] 
    
    nimgs = []
    
    N = bboxs.shape[0]

    for i in range(N):
        bbox = bboxs[i]
        bbox = bbox[0:4]

        #bbx selection
        #bbw = bbox[:, 2] - bbox[:, 0] + 1
        #bbh = bbox[:, 3] - bbox[:, 1] + 1
        #area = bbw * bbh
        #ind = np.where(area == np.max(area))
        #bbox = bbox[ind[0][0],0:4]
        ###
        point = points[i]
        point = point.reshape((2,5)).T
        #print(bbox)
        #print(points)
        nimg = face_preprocess.preprocess_out_128(face_img, bbox, point, image_size='112,112')
        #nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        #aligned = np.transpose(nimg, (2,0,1))
        nimgs.append(nimg) 
    return nimgs #aligned


  def get_input_out_112(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    #bbox = bbox[0,0:4]
    #bbx selection
    bbw = bbox[:, 2] - bbox[:, 0] + 1
    bbh = bbox[:, 3] - bbox[:, 1] + 1
    area = bbw * bbh
    ind = np.where(area == np.max(area))
    bbox = bbox[ind[0][0],0:4]
    ###
    #points = points[0,:].reshape((2,5)).T
    points = points[ind[0][0],:].reshape((2,5)).T
    #print(bbox)
    #print(points)
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    #nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    #aligned = np.transpose(nimg, (2,0,1))
    return nimg #aligned



  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()

    return embedding

  def get_feature_batch(self, aligned):
    if(len(aligned.shape) == 3):
        return get_feature(aligned)
    else:
        input_blob = aligned 

    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()

    return embedding


  def get_ga(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.ga_model.forward(db, is_train=False)
    ret = self.ga_model.get_outputs()[0].asnumpy()
    g = ret[:,0:2].flatten()
    gender = np.argmax(g)
    a = ret[:,2:202].reshape( (100,2) )
    a = np.argmax(a, axis=1)
    age = int(sum(a))

    return gender, age

