import os
import os.path as osp


prefix = '/data1/chai/anti-spoof/'
lst_set = ['train.csv', 'val.csv']

label_dict = {'real':0, 'spoof':1}
for lst in lst_set:
  file_lst = lst.replace('.csv', '.lst')
  fp = open(file_lst, 'w')
  for line in open(lst).readlines():
    wds = line.strip().split(',') 
    fname,label = wds[0],wds[1]
    label = label_dict[label]
    fname = osp.join(prefix, fname)
    fp.write('%s %d\n'%(fname, label))
  fp.close()
