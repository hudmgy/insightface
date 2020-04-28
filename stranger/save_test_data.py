import numpy as np
import os


fs = open('similary.txt', 'w')
ff = open('featu.txt', 'w')
xdat = np.load('xdat.npy')
for x in xdat:
  for d in x[:512]:
    ff.write('%f '%d)
  ff.write('\n')
  for d in x[512:]:
    fs.write('%f '%d)
  fs.write('\n')


fs.close()
ff.close()
