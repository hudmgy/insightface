import os
import os.path as osp
import shutil as sh
import random


base_path = '20190725'
selec_num = 40

Ids = []
for fold in os.listdir(base_path):
  folder = osp.join(base_path, fold)
  if osp.isdir(folder):
    Ids.append(folder)

del_ids = random.sample(range(len(Ids)), len(Ids)-selec_num)
for i in del_ids:
  print(Ids[i])
  sh.rmtree(Ids[i])
