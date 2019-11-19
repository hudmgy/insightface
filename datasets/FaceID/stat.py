import os
import os.path as osp
from collections import defaultdict
import random
import ipdb


root='train/ms1m/'
train_set = open('train_10000.txt','w')
val_set = open('val_10000.txt','w')
train_set_ex = open('train_1000.txt','w')
val_set_ex = open('val_1000.txt','w')
train_unlabeled = open('train_unlabeled.txt', 'w')
val_unlabeled = open('val_unlabeled.txt', 'w')
train_num = 10000
ex_train_num = 1000


stat = defaultdict(int)
filetree = []


def length(elem):
  return len(elem)


for idx, fold in enumerate(os.listdir(root)):
  folder = osp.join(root, fold)
  files = []
  for fi in os.listdir(folder): 
    if fi.endswith('.jpg'):
      files.append(osp.join(folder, fi))
  filetree.append(files)
  stat[fold] = len(files)

#for k,v in stat.items():
#  print(k, v)

id_cts = 0
filetree.sort(key=length, reverse=True)
for files in filetree:
  if len(files) > 140: 
    continue
  if len(files) < 20:
    if random.randint(0,9)==0:
      for fi in files:
        val_unlabeled.write('%s unkown\n'%(fi))
    else:
      for fi in files:
        train_unlabeled.write('%s unkown\n'%(fi))
    continue

  if id_cts < train_num:
    random.shuffle(files)
    for idx,fi in enumerate(files):
      if idx > 7:
        train_set.write('%s %d\n'%(fi, id_cts))
      else:
        val_set.write('%s %d\n'%(fi, id_cts))
    id_cts += 1
  elif id_cts < train_num + ex_train_num:
    random.shuffle(files)
    for idx,fi in enumerate(files):
      if idx > 7:
        train_set_ex.write('%s %d\n'%(fi, id_cts))
      else:
        val_set_ex.write('%s %d\n'%(fi, id_cts))
    id_cts += 1
  

train_set.close()
val_set.close()
train_set_ex.close()
val_set_ex.close()
train_unlabeled.close()
val_unlabeled.close()
