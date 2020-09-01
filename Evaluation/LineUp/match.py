import os, sys
import numpy as np
import os.path as osp
import time
import shutil as sh
from collections import defaultdict
import ipdb

g_file_buff = []

def load_dataset(folder):
    dataset = defaultdict(dict)
    for dirname, _, files in os.walk(folder):
        for fi in files:
            if fi.endswith('.npy'):
                frame_no = fi.split('_')[0]
                filename = osp.join(dirname, fi)
                vec = np.load(filename)
                if frame_no not in dataset.keys():
                    dataset[frame_no]['file'] = []
                    dataset[frame_no]['featu'] = []
                # vec = vec / np.linalg.norm(vec)
                dataset[frame_no]['file'].append(filename)
                dataset[frame_no]['featu'].append(vec)
    print('frame count: ', len(dataset))

    for k in dataset.keys():
        dataset[k]['featu'] = np.array(dataset[k]['featu'])
    return dataset

def parse_fname(fname):
    fname = osp.splitext(osp.basename(fname))[0]
    frame_no, x, y = fname.split('_')
    frame_no = int(frame_no)
    x = int(x[1:])
    y = int(y[1:])
    return frame_no, x, y


def match(data_gallery, data_probe, threshold=0.5):
    featu_g = data_gallery['featu']
    featu_p = data_probe['featu']
    file_g = data_gallery['file']
    file_p = data_probe['file']

    mmap = featu_p.dot(featu_g.T)
    col_max = np.argsort(mmap,0)[-1,:]
    col_pairs = [(col_max[i], i) for i in range(len(col_max))]
    row_max = np.argsort(mmap,1)[:,-1]
    row_pairs = [(i, row_max[i]) for i in range(len(row_max))]

    cross_matched_pairs = []
    for d in col_pairs:
        if d in row_pairs and mmap[d] > threshold:
            pf = file_p[d[0]]
            gf = file_g[d[1]]
            p_frame_no, p_x, p_y = parse_fname(pf)
            g_frame_no, g_x, g_y = parse_fname(gf)

            if abs(p_frame_no - g_frame_no) < 750: continue
            if abs(p_y - g_y) < 200: continue
            if pf in g_file_buff or gf in g_file_buff: continue

            pair = {}
            pair['id'] = d
            pair['file'] = (pf, gf)
            pair['frame_no'] = (p_frame_no, g_frame_no)
            pair['coord'] = ((p_x, p_y), (g_x, g_y))
            pair['similary'] = mmap[d]

            g_file_buff.append(pf)
            g_file_buff.append(gf)
            cross_matched_pairs.append(pair)
    return cross_matched_pairs


def save_matched_pairs(cross_matched_pairs, save_dir):
    for i, pair in enumerate(cross_matched_pairs):
        folder = osp.join(save_dir, '%06d_%.3f'%(i, pair['similary']))
        if not osp.exists(folder):
            os.makedirs(folder)
        pf, gf = pair['file']
        pf = pf.replace('.npy', '.jpg')
        gf = gf.replace('.npy', '.jpg')
        dst_pf = osp.join(folder, osp.basename(pf))
        dst_gf = osp.join(folder, osp.basename(gf))
        sh.copy(pf, dst_pf)
        sh.copy(gf, dst_gf)


if __name__=='__main__':
    data_dir = '/home/ct/Data/queue/crops'
    save_dir = '/home/ct/Data/queue/matched'

    dataset = load_dataset(data_dir)
    fn_list = list(dataset.keys())
    fn_list.sort()
    cross_matched_pairs = []
    for i in range(len(fn_list))[1:]:
        data_p = dataset[fn_list[i]]
        for j in range(i):
            data_g = dataset[fn_list[j]]
            # ipdb.set_trace()
            cross_matched_pairs += match(data_g, data_p, threshold=0.75)
    # ipdb.set_trace()
    save_matched_pairs(cross_matched_pairs, save_dir)

