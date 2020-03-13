import face_model
import argparse
import cv2
import sys
import numpy as np

import os

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
#parser.add_argument('--model', default='../recognition/models/model-r100-Insightface-faces_ms1m_emore_sany_hik_ziyan_20190916/r100-arcface-ms1m_emore_sany_hik_ziyan_20190916/model,1', help='path to load model.')

#parser.add_argument('--model', default='../recognition/models/model-r100-Insightface-faces_ms1m_emore_sany_hik_ziyan_20191113/r100-arcface-ms1m_emore_sany_hik_ziyan_20191113/model,1', help='path to load model.')

parser.add_argument('--model', default='../recognition/models/model-r100-Insightface-faces_ms1m_emore_sany_hik_ziyan_20190916_0.943_0.7854/r100-arcface-ms1m_emore_sany_hik_ziyan_20190916/model,1', help='path to load model.')
#parser.add_argument('--model', default='../recognition/models/model-r100-Insightface-faces_ms1m_emore_sany_hik_ziyan_with_color_cutoff_aug_20190820/r100-arcface-ms1m_emore_sany_hik_ziyan_20190802/model,1', help='path to load model.')
parser.add_argument('--ga-model', default='../models/gamodel-r50/model,0', help='path to load model.')
parser.add_argument('--gpu', default=1, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

'''
model = face_model.FaceModel(args)
img = cv2.imread('21017903/21017903_44.jpg')
img = model.get_input(img)
f1 = model.get_feature(img)
print(f1[0:10])
#gender, age = model.get_ga(img)
#print(gender)
#print(age)
#sys.exit(0)
img = cv2.imread('21017903/21017903_3.jpg')
img = model.get_input(img)
f2 = model.get_feature(img)
print(f2[1:10])
dist = np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
'''
#test the model on sany data
#the data including 1. prob/query data, gallery data, distractor data, distractor data can use megaface,
#prob/query can use the one day surveilence data, gallery data use the registritation data of all the stuff

def extract_featues_of_images(img_list, save_path):
    
    #img_list = "/data2/TZ/face_data/face_sany/Register/gallery.list"
    
    fid = open(img_list, "r")
    lines = fid.readlines()
    fid.close()
    
    fid = open(save_path, "w")
    
    model = face_model.FaceModel(args)
    folder_cropped_images = "./cropped_prob/"
    
    
    for ind, line in enumerate(lines):
        
        line = line.strip()
        print("%d, %s" %(ind,line))
        img = cv2.imread(line)
        if(img is None):
            continue
        img1 = model.get_input(img)
        
        if( img1 is None):
            continue

        print("img1.size = %d/%d/%d\n" % (img1.shape[0], img1.shape[1], img1.shape[2]))
        
        name = line.split("/")[-1]
        #cv2.imwrite(folder_cropped_images + name, np.transpose(img1, (1,2,0)))

        f1 = model.get_feature(img1)
        name = line.split("/")[-1].split(".")[0]
        fid.write(name + ",")
        #fid.write(name + ",")
        for inx, fea in enumerate(f1):
            if(inx == 511):
                fid.write(str(fea) + "\n")
            else:
                fid.write(str(fea) + " ")
    fid.close()

def extract_featues_of_images_only_recog(img_list, save_path):
    
    #img_list = "/data2/TZ/face_data/face_sany/Register/gallery.list"
    
    fid = open(img_list, "r")
    lines = fid.readlines()
    fid.close()
    
    fid = open(save_path, "w")
    
    model = face_model.FaceModel(args)
    folder_cropped_images = "./cropped_prob/"
    
    
    for ind, line in enumerate(lines):
        
        line = line.strip()
        print("%d, %s" %(ind,line))
        img = cv2.imread(line)
        img1 = np.transpose(img, (2,0,1))
        print("img.size = %d/%d" % (img1.shape[0], img1.shape[1]) )
	
        f1 = model.get_feature(img1)
        name = line.split("/")[-1].split(".")[0]
        fid.write(line + ",")
        #fid.write(name + ",")
        for inx, fea in enumerate(f1):
            if(inx == 511):
                fid.write(str(fea) + "\n")
            else:
                fid.write(str(fea) + " ")

    fid.close()

def extract_featues_of_images_only_recog_112_batch(img_list, save_path):
    
    #img_list = "/data2/TZ/face_data/face_sany/Register/gallery.list"
    
    fid = open(img_list, "r")
    lines = fid.readlines()
    fid.close()
    
    fid = open(save_path, "w")
    
    model = face_model.FaceModel(args)
    folder_cropped_images = "./cropped_prob/"
    

    batch = 240 

    np_batch = np.zeros((batch, 3, 112, 112)) 

    num = -1 

    N = len(lines)
    batch_lines = [] 

    start_pt = 0
    
    for ind, line in enumerate(lines):
        num += 1 

        line = line.strip()
        print("%d, %s" %(ind,line))
        img = cv2.imread(line)

        #img = img[8:120, 8:120, :] 
        
        img1 = np.transpose(img, (2,0,1))
        if(start_pt + batch < N-1): 
            if(num < batch):
                np_batch[num, :, :, :] = img1
                batch_lines.append(line)
                print("img.size = %d/%d" % (img1.shape[0], img1.shape[1]) )
            else:
                f1 = model.get_feature_batch(np_batch)
                f1 = f1.reshape((batch, 512))
                for i in range(batch):
                    fid.write(batch_lines[i] + ",")
                    for inx, fea in enumerate(f1[i, :]):
                        if(inx == 511):
                            fid.write(str(fea) + "\n")
                        else:
                            fid.write(str(fea) + " ")

                start_pt = ind
                num = 0
                batch_lines = []
                np_batch[num, :, :, :] = img1
                batch_lines.append(line)
                print("img.size = %d/%d" % (img1.shape[0], img1.shape[1]) )

        else:
            f1 = model.get_feature(img1)
            #name = line.split("/")[-1].split(".")[0]
            fid.write(line + ",")
            #fid.write(name + ",")
            for inx, fea in enumerate(f1):
                if(inx == 511):
                    fid.write(str(fea) + "\n")
                else:
                    fid.write(str(fea) + " ")

    fid.close()



def compare_features(gallery_list, prob_list):
    fid = open(gallery_list, "r")
    lines = fid.readlines()
    fid.close()
    gallery_feas={}
    for ind,line in enumerate(lines):
        print(ind)
        line = line.strip()
        splits = line.split(",")
        name = splits[0]
        lists = splits[1].split()
        feas = [float(fea) for fea in lists]
        gallery_feas[name] = feas

    fid = open(prob_list, "r")
    lines = fid.readlines()
    fid.close()
    prob_feas={}
    for ind,line in enumerate(lines):
        print(ind)
        line = line.strip()
        splits = line.split(",")
        name = splits[0]
        lists = splits[1].split()
        feas = [float(fea) for fea in lists]
        prob_feas[name] = feas
    
    
    num_correct = 0
    correct_fid = open("correct.list", "w")
    wrong_fid = open("wrong.list", "w")
    max_val_fid = open("max_val.list", "w")

    num = 0

    for (prob_name, prob_val) in prob_feas.items():
        max_name =""
        max_val = 0
        num = num + 1
        print(num)
        for(gal_name, gal_val) in gallery_feas.items():
            val = np.dot(gal_val, prob_val)
            if(val > max_val):
                max_val = val
                max_name = gal_name
        max_val_fid.write(str(max_val) + "\n")

        if(max_val >= 0.0):
            p_name = prob_name.split("_")[0]
            if(p_name == max_name):
                num_correct = num_correct + 1
                correct_fid.write(prob_name + "\n")
            else:
                wrong_fid.write(prob_name + "\n")
        else:
            wrong_fid.write(prob_name + "\n")

    correct_ratio = float(num_correct) / (len(prob_feas))
    print(correct_ratio)

    correct_fid.close()
    wrong_fid.close()
    max_val_fid.close()


def compare_features_test_new_model(gallery_list, prob_list):
    fid = open(gallery_list, "r")
    lines = fid.readlines()
    fid.close()
    gallery_feas={}
    for ind,line in enumerate(lines):
        if(ind %1000 == 0):
            print("gallery_fea: ", ind)
        line = line.strip()
        splits = line.split(",")
        name = splits[0]
        lists = splits[1].split()
        feas = [float(fea) for fea in lists]
        gallery_feas[name] = feas

    fid = open(prob_list, "r")
    lines = fid.readlines()
    fid.close()
    prob_feas={}
    for ind,line in enumerate(lines):
        if(ind %1000 == 0):
            print("prob_fea: ", ind)
        line = line.strip()
        splits = line.split(",")
        name = splits[0]
        lists = splits[1].split()
        feas = [float(fea) for fea in lists]
        prob_feas[name] = feas
    
    
    num_correct = 0
    correct_fid = open("correct.list", "w")
    wrong_fid = open("wrong.list", "w")
    max_val_fid = open("max_val.list", "w")

    num = 0

    for (prob_name, prob_val) in prob_feas.items():
        max_name =""
        max_val = 0
        num = num + 1
        print("compare: ", num)
        for(gal_name, gal_val) in gallery_feas.items():
            val = np.dot(gal_val, prob_val)
            if(val > max_val):
                max_val = val
                print("gal_name: ", gal_name)    
                if(not "-" in gal_name):
                    continue
                max_name = gal_name.split("-")[1]
        max_val_fid.write(str(max_val) + "\n")

        if(max_val >= 0.0):
            p_name = prob_name.split("_")[0]
            if(p_name == max_name):
                num_correct = num_correct + 1
                correct_fid.write(prob_name + "\n")
            else:
                wrong_fid.write(prob_name + "\n")
        else:
            wrong_fid.write(prob_name + "\n")

    correct_ratio = float(num_correct) / (len(prob_feas))
    print(correct_ratio)

    correct_fid.close()
    wrong_fid.close()
    max_val_fid.close()

def compare_features_test_new_model_with_numpy(gallery_list, prob_list):
    fid = open(gallery_list, "r")
    lines = fid.readlines()
    fid.close()
    gallery_feas={}
    for ind,line in enumerate(lines):
        if(ind % 1000 == 0):
            print("gallery_fea: ", ind)
        line = line.strip()
        splits = line.split(",")
        name = splits[0]
        lists = splits[1].split()
        feas = [float(fea) for fea in lists]
        gallery_feas[name] = feas

    fid = open(prob_list, "r")
    lines = fid.readlines()
    fid.close()
    prob_feas={}
    for ind,line in enumerate(lines):
        if(ind % 1000 == 0):
            print("prob_fea: ", ind)
        line = line.strip()
        splits = line.split(",")
        name = splits[0]
        lists = splits[1].split()
        feas = [float(fea) for fea in lists]
        prob_feas[name] = feas
    
    
    num_correct_larger_05 = 0
    num_correct = 0
    correct_fid = open("correct.list.self_model_numpy_old_diku", "w")
    wrong_fid = open("wrong.list.sefl_model_numpy_old_diku", "w")
    max_val_fid = open("max_val.list.self_model_numpy_old_diku", "w")

    num = 0

    test_num = len(prob_feas.items())
    gal_num = len(gallery_feas.items())
    
    np_matrix_test = np.zeros((test_num, 512))
    np_matrix_gal = np.zeros((gal_num, 512))

    for (prob_name, prob_val) in prob_feas.items():
        np_matrix_test[num, :] = prob_val
        num += 1
    
    num = 0
    for(gal_name, gal_val) in gallery_feas.items():
        np_matrix_gal[num, :] = gal_val
        num += 1
    
    np_matrix_gal = np.transpose(np_matrix_gal)

    np_val = np.dot(np_matrix_test, np_matrix_gal)
    
    max_val_list = np.max(np_val, axis = 1)
    max_val_tmp = max_val_list.tolist()

    max_val_list = max_val_list.reshape((test_num, 1))
    mm = np.where(np_val == max_val_list)
    mm = mm[1]
    mm = mm.tolist()

    gal_keys = gallery_feas.keys()

    max_val_list = max_val_tmp 
     
    num = 0
    for (prob_name, prob_val) in prob_feas.items():
        max_name =""
        max_val = 0
        print("compare: ", num)
        
        max_name = gal_keys[mm[num]]

        if(not "-" in max_name):
            num = num + 1
            continue

        max_name = max_name.split("-")[1].split(".")[0]

        max_val = max_val_list[num]

        num = num + 1
        max_val_fid.write(str(max_val) + "\n")

        if(max_val >= 0.0):
            p_name = prob_name.split("/")[-1].split("_")[1]
            if(p_name == max_name):
                num_correct = num_correct + 1
                if(max_val >= 0.5):
                    num_correct_larger_05 += 1
                correct_fid.write(prob_name + "\n")
            else:
                wrong_fid.write(prob_name + "\n")
        else:
            wrong_fid.write(prob_name + "\n")

    correct_ratio = float(num_correct) / (len(prob_feas))
    print(correct_ratio)

    correct_ratio_larger_05 = float(num_correct_larger_05) / (len(prob_feas))
    print(correct_ratio_larger_05)

    correct_fid.close()
    wrong_fid.close()
    max_val_fid.close()




def compare_features_lfw(gallery_list, prob_list):
    fid = open(gallery_list, "r")
    lines = fid.readlines()
    fid.close()
    gallery_feas={}
    for ind, line in enumerate(lines):
        print(ind)
        if(ind == 3673):
            xxx = 0
            line = line.strip()
            splits = line.split(",")
            print(len(splits))
            name = splits[0]
            lists = splits[1].split()
            feas = [float(fea) for fea in lists]
            gallery_feas[name] = feas

    fid = open(prob_list, "r")
    lines = fid.readlines()
    fid.close()
    prob_feas={}
    for line in lines:
        line = line.strip()
        splits = line.split(",")
        name = splits[0]
        lists = splits[1].split()
        feas = [float(fea) for fea in lists]
        prob_feas[name] = feas
    
    
    num_correct = 0
    correct_fid = open("correct.list", "w")
    wrong_fid = open("wrong.list", "w")
    max_val_fid = open("max_val.list", "w")

    num = 0

    for (prob_name, prob_val) in prob_feas.items():
        max_name =""
        max_val = 0
        num = num + 1
        print(num)
        for(gal_name, gal_val) in gallery_feas.items():
            val = np.dot(gal_val, prob_val)
            if(val > max_val):
                max_val = val
                max_name = gal_name
        max_val_fid.write(str(max_val) + "\n")

        if(max_val >= 0.0):
            p_name = prob_name[:-9]
            max_name = max_name[:-9]
            if(p_name == max_name):
                num_correct = num_correct + 1
                correct_fid.write(prob_name + "\n")
            else:
                wrong_fid.write(prob_name + "\n")
        else:
            wrong_fid.write(prob_name + "\n")

    correct_ratio = float(num_correct) / (len(prob_feas))
    print(correct_ratio)

    correct_fid.close()
    wrong_fid.close()
    max_val_fid.close()

               
###extrack the face features
#img_list = "/data2/TZ/face_data/face_sany/Register/gallery.list"
#save_path = "/data2/TZ/face_data/face_sany/Register/gallery.fea"
#img_list = "/data/TZ/face_sany/Register/prob_sany_309.list"
#save_path = "/data/TZ/face_sany/Register/prob_sany_309.fea"
#img_list = "./lfw_gallery.list"
#save_path = "./lfw_gallery.fea"
#img_list = "./lfw_prob.list"
#save_path = "./lfw_prob.fea"


#img_list = "/data2/TZ/face_data/huang/selected.list"
#ksave_path = "/data2/TZ/face_data/huang/selected.fea"

#img_list = "/data/TZ/face/test_data/diku.list"
#save_path = "/data/TZ/face/test_data/diku.fea"

#img_list = "/data/TZ/face/test_data/diku/diku_new.list"
#save_path = "/data/TZ/face/test_data/diku_new_model.fea"


img_list = "/data/TZ/face/test_data/test_img_sany_filtered.list"
save_path = "/data/TZ/face/test_data/test_img_sany_filtered_new_model.fea"



#extract_featues_of_images_only_recog(img_list, save_path)
#extract_featues_of_images(img_list, save_path)



##get the accuracy
#prob_fea ="/data/TZ/face_sany/Register/prob_sany_309.fea"
#gal_fea ="/data/TZ/face_sany/Register/gallery.fea"
#prob_fea ="./lfw_prob.fea"
#gal_fea ="./lfw_gallery.fea"

#prob_fea ="/data/TZ/insightface/recognition/models/r100-arcface-ms1m_emore_sany_hik_ziyan_20190802.complete_83/test_img_sany_self.fea"
#gal_fea ="/data/TZ/insightface/recognition/models/r100-arcface-ms1m_emore_sany_hik_ziyan_20190802.complete_83/diku.fea"
#
#prob_fea ="/data/TZ/insightface/recognition/models/model-r100-Insightface-faces_ms1m_emore_sany_hik_ziyan_with_color_cutoff_aug_20190820_83/test_img_sany.fea"
#gal_fea ="/data/TZ/insightface/recognition/models/model-r100-Insightface-faces_ms1m_emore_sany_hik_ziyan_with_color_cutoff_aug_20190820_83/diku.fea"
#
#prob_fea ="/data/TZ/insightface/recognition/models/r100-arcface-ms1m_emore_sany_hik_ziyan_20190802/test_img_sany_self.fea"
#gal_fea ="/data/TZ/insightface/recognition/models/r100-arcface-ms1m_emore_sany_hik_ziyan_20190802/diku.fea"


#compare_features(gal_fea, prob_fea)
#compare_features_test_new_model_with_numpy(gal_fea, prob_fea)



prob_img_list = "/data/chaizh/ID_Test/test_img_sany_filtered_aligned.list"
prob_fea = "/data/chaizh/ID_Test/prob_img.fea"
gal_img_list = "/data/chaizh/ID_Test/diku_alingned.list"
gal_fea = "/data/chaizh/ID_Test/gal_img.fea"

extract_featues_of_images_only_recog_112_batch(prob_img_list, prob_fea)
extract_featues_of_images_only_recog_112_batch(gal_img_list, gal_fea)

compare_features_test_new_model_with_numpy(gal_fea, prob_fea)











'''
model = face_model.FaceModel(args)
print(f1[0:10])
#gender, age = model.get_ga(img)
#print(gender)
#print(age)
#sys.exit(0)
img = cv2.imread('Tom_Hanks_54745.png')
img = model.get_input(img)
f2 = model.get_feature(img)
print(f2[1:10])
dist = np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
'''


