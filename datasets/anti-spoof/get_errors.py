import os
import os.path as osp
import shutil as sh
import ipdb


pos_n = 0
neg_n = 0
fr = 0
fa = 0

filename = 'pred.csv'
Th = 0.1

with open(filename) as fp:
  for line in fp.readlines():
    fi,gt,score = line.strip().split(',')

    gt = int(gt)
    score = float(score)
    check_flag = 'clear_ok'
    if gt==1: 
        pos_n += 1
        if score < Th:
            check_flag = 'noise'
            fr += 1
    elif gt==0:
        neg_n += 1
        if score > Th:
            check_flag = 'noise'
            fa += 1

    #raw_file = fi.replace('crop128x128', 'crop144x144')
    raw_file = fi
    dst_file = fi.replace('crop128x128', 'results/'+check_flag)
    fname,ext = osp.splitext(dst_file)
    dst_file = '%s_%.3f%s'%(fname, score, ext)
    print(raw_file, dst_file)

    if not osp.exists(osp.dirname(dst_file)):
      os.makedirs(osp.dirname(dst_file))
    sh.copy(raw_file, dst_file)

print(fr, pos_n, float(fr)/pos_n)
print(fa, neg_n, float(fa)/neg_n)
