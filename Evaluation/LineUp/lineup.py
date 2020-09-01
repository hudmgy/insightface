import os, sys
import numpy as np
import datetime
import glob
import os.path as osp
from skimage import transform as trans
from retinaface import RetinaFace
from collections import defaultdict
import cv2
import ipdb

from det_crop import detect, filter_outline_faces, save_face_coord
from det_crop import align_crop, save_image, drawing, gen_valid_region
from gen_features import get_feature, load_faceid_model


def distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    L2 = p1 - p2
    L2 = L2.dot(L2)
    L2 = np.sqrt(L2)
    return L2

def map_coord2num(faces, frame_no, thresh=200):
    coord = []
    for i in range(faces.shape[0]):
        box = faces[i].astype(np.int)
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        coord.append([cx, cy])
    num = len(coord)
    if num > 0:
        coord = np.array(coord)
        index = np.argsort(coord[:,0])
        coord = coord[index]
    
    # ###
    coord2num = []
    for i in range(1, num):
        for j in range(i):
            p1, p2 = coord[j], coord[i]
            L2 = distance(p1, p2)
            if L2 > thresh:
                dn = i - j
                coord2num.append(list(p1) + list(p2) + [dn, frame_no])
    return coord2num


def clear_coord2num(coord2num_mapping, timeout_frame_no):
    i = 0
    for idx, mp in enumerate(coord2num_mapping):
        if mp[-1] >= timeout_frame_no:
            i = idx
            break
    return coord2num_mapping[i:]


def search_knn_in_mapping(coord2num_mapping, ppair, k=3):
    mapping = np.array(coord2num_mapping)[:,:4]
    ppair = np.array(ppair)
    d_mapping = mapping - ppair
    d2_mapping = d_mapping * d_mapping
    d2_mapping = d2_mapping.sum(1)
    idx = np.argsort(d2_mapping)
    nnp = [coord2num_mapping[i][4] for i in idx[:k]]
    mean_num = np.mean(np.array(nnp))
    return mean_num, idx[:k]


def clear_face_buffer(face_buffer, timeout_frame_no):
    i = 0
    for idx, bf in enumerate(face_buffer):
        if bf['frame_no'] >= timeout_frame_no:
            i = idx
            break
    return face_buffer[i:]


def pairing(data_gallery, data_probe, id_thresh=0.5, l2_thresh=200):
    g_frame_no = data_gallery['frame_no']
    p_frame_no = data_probe['frame_no']
    if abs(p_frame_no - g_frame_no) < 750: 
        return []

    featu_g = data_gallery['featu']
    featu_p = data_probe['featu']
    image_g = data_gallery['image']
    image_p = data_probe['image']
    box_g = data_gallery['box']
    box_p = data_probe['box']
    paired_g = data_gallery['paired']
    paired_p = data_probe['paired']

    mmap = featu_p.dot(featu_g.T)
    col_max = np.argsort(mmap,0)[-1,:]
    col_pairs = [(col_max[i], i) for i in range(len(col_max))]
    row_max = np.argsort(mmap,1)[:,-1]
    row_pairs = [(i, row_max[i]) for i in range(len(row_max))]

    cross_matched_pairs = []
    for d in col_pairs:
        if d in row_pairs and mmap[d] > id_thresh:
            p1 = box_p[d[0]][:2]
            p2 = box_g[d[1]][:2]
            if paired_p[d[0]] is True or paired_g[d[1]] is True:
                continue
            if distance(p1, p2) < l2_thresh:
                continue

            pair = {}
            pair['id'] = d[0]
            pair['image'] = (image_p[d[0]], image_g[d[1]])
            pair['frame_no'] = (p_frame_no, g_frame_no)
            pair['coord'] = p1 + p2
            pair['similary'] = mmap[d]
            cross_matched_pairs.append(pair)
    return cross_matched_pairs



if __name__ == '__main__':
    thresh = 0.2
    scales = [1440, 2560]
    thresh2 = 0.9
    scales2 = [1024, 1980]
    factor = 1.50

    vid_path = sys.argv[1]
    gpuid = int(sys.argv[2])
    en_drawing = True
    coord2num_mapping = []
    face_buffer = []
    waiting_time = []
    num_in_queue = []
    mean_num_in_queue = 0
    mean_waiting_time = 'null'
    text1 = 'queuing length: null'
    text2 = 'waiting time: null'
    faces = []
    landmarks = []

    # detector
    detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
    # feature extractor
    feature_nets = load_faceid_model('model-r50/model,0', gpuid=0, image_size='3,112,112')
    # setting roi
    border_lines, point_pair = gen_valid_region()

    cap = cv2.VideoCapture(vid_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 如果是avi视频，编码需要为MJPG
    vwriter = cv2.VideoWriter('demo.mp4', fourcc, 1, (2560//2, 1440//2))

    frame_no = -1
    while(True):
        ret, frame = cap.read()
        if ret==False:
            break
        frame_no += 1
        if frame_no%50==0:
            print(frame_no)
            # 排队人数
            faces, landmarks = detect(detector, scales, frame, thresh)
            faces, landmarks = filter_outline_faces(faces, landmarks, border_lines)
            num_in_queue.append(len(faces))

            # 更新位置-人数映射表
            timeout_frame_no = max(frame_no - 15000, 0)
            coord2num_mapping += map_coord2num(faces, frame_no, thresh=200)
            oord2num_mapping = clear_coord2num(coord2num_mapping, timeout_frame_no)
 
            # 寻找同人不同位置的记录
            faces2, landmarks2 = detect(detector, scales2, frame, thresh2)
            faces2, landmarks2 = filter_outline_faces(faces2, landmarks2, border_lines)
            crop_imgs, crop_boxes = align_crop(frame, faces2, landmarks2, min_size=60)
            face_buffer = clear_face_buffer(face_buffer, timeout_frame_no)
            if len(crop_imgs) > 0:
                cross_matched_pairs = []
                num_time_pair = []
                features = get_feature(crop_imgs, feature_nets)
                curr_buffer = {}
                curr_buffer['frame_no'] = frame_no
                curr_buffer['featu'] = features
                curr_buffer['image'] = crop_imgs
                curr_buffer['box'] = crop_boxes
                curr_buffer['paired'] = [False] * len(crop_boxes)
                for buffer in face_buffer:
                    cross_matched = pairing(buffer, curr_buffer, id_thresh=0.81, l2_thresh=300)
                    for p in cross_matched:
                        curr_buffer['paired'][p['id']] = True
                    cross_matched_pairs += cross_matched
                for pair in cross_matched_pairs:
                    pass_num,_ = search_knn_in_mapping(coord2num_mapping, pair['coord'], k=3)
                    num_time_pair.append([pass_num, (pair['frame_no'][0]-pair['frame_no'][1])])
                face_buffer.append(curr_buffer)
                print('num_time_pair:', num_time_pair)

            # waiting time
            mean_num_in_queue = np.mean(np.array(num_in_queue[-10:]))
            for ntp in num_time_pair:
                curr_v = (ntp[1] / ntp[0]) / 25
                waiting_time.append(curr_v)
            if len(waiting_time) > 0:
                mean_waiting_time = np.mean(np.array(waiting_time[-200:]))
                
            text1 = 'queuing length: %d'%(int(mean_num_in_queue))
            if mean_waiting_time!='null':
                mean_waiting_time *= mean_num_in_queue
                text2 = 'waiting time: %dmin%ds'%(int(mean_waiting_time)//60, int(mean_waiting_time)%60)
            print(text1)
            print(text2)
            # display
            if en_drawing:
                dis_img = drawing(frame, faces, landmarks)
                for bl in point_pair:
                    cv2.line(dis_img, (bl['x1'], bl['y1']),
                                (bl['x2'], bl['y2']), (255, 0, 0), 6)
                cv2.putText(dis_img, text1, (800, 100),
                                cv2.FONT_HERSHEY_COMPLEX, 3.0, (0, 0, 255), 5)
                cv2.putText(dis_img, text2, (800, 200),
                                cv2.FONT_HERSHEY_COMPLEX, 3.0, (0, 0, 250), 5)
                # drawing_file = osp.join(osp.splitext(vid_path)[0], '%06d.jpg'%frame_no)
                # save_image(drawing_file, dis_img)
                dis_img = cv2.resize(dis_img, (2560//2, 1440//2))
                vwriter.write(dis_img)
    vwriter.release()
