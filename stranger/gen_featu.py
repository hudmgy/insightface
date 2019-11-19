import os
import numpy as np
import os.path as osp
import ipdb


data_path = '/data1/chai/strangers/featus'
featu_path = '../datasets/stranger/featu_train'
featu_files = ['sany_diku-caps20191022.txt', 'sany_diku-Sidewalk.txt']




def load_featu(ff, thres_h=0.6, thres_l=0.25):
  in_featu = []
  out_featu = []
  with open(ff) as fp:
    for line in fp.readlines():
      line = line.strip()
      wds = line.split(',')
      top_s = float(wds[2])
      if top_s > thres_h or top_s < thres_l:
        continue
      if wds[1]=='in':
        in_featu.append(np.array(wds[2:]))
      elif wds[1]=='out':
        out_featu.append(np.array(wds[2:]))
  return in_featu, out_featu


in_featu_bin = []
out_featu_bin = []
for ff in featu_files:
  in_featu, out_featu = load_featu(osp.join(data_path, ff))  
  #ipdb.set_trace()
  in_featu_bin.extend(in_featu)
  out_featu_bin.extend(out_featu)

in_featu_bin = np.array(in_featu_bin)
out_featu_bin = np.array(out_featu_bin)
if not osp.exists(featu_path):
  os.makedirs(featu_path)
np.save(osp.join(featu_path,'in_featu.npy'), in_featu_bin)
np.save(osp.join(featu_path,'out_featu.npy'), out_featu_bin)
