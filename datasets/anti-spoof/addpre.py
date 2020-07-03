import os
import os.path as osp


prefix = '/data/chaizh/anti-spoof-0430/'
lst_set = ['train.csv', 'test.csv']
#prefix = '/data/chaizh/anti-spoof-0430/crop224x224'
#lst_set = ['real_test.csv']


for lst in lst_set:
  file_lst = lst.replace('.csv', '.lst')
  fp = open(file_lst, 'w')
  lst = osp.join(prefix, lst)
  for line in open(lst).readlines():
    wds = line.strip().split(',') 
    fname,label = wds[0],wds[1]
    fname = osp.join(prefix, fname)
    fp.write('%s,%s\n'%(fname, label))
  fp.close()
