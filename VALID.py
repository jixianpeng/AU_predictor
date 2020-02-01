
print('loading model.....')
# from model_with_cat_boder_landmark import *
# from model_for_fanhua import *
from Aspect_Based_Fusion import *
import numpy as np
from PIL import Image
import torch
import os
import json
import cv2
cc = ResNet().cuda().eval()
ss = Landmark_diff_module().cuda().eval()
aq = Aspect_Query().cuda().eval()
re_training=False
old_static_state = torch.load('./cc')
cc.load_state_dict(old_static_state)
old_static_state = torch.load('./ss')
ss.load_state_dict(old_static_state)
old_static_state = torch.load('./aq')
aq.load_state_dict(old_static_state)
print("*"*10+'model already'+"*"*10)


print('loading data.....')
boder_path='D:/DATASET/aff_wild2/crop and aligned/V_S_boder/'
landmark_path='D:/DATASET/aff_wild2/crop and aligned/V_landmark/'
seg_path='D:/DATASET/aff_wild2/crop and aligned/V_S/'
label_path='D:\DATASET/aff_wild2/annotations/AU_Set/Validation_Set/'

landmarks_by_file={}
for i in os.listdir(landmark_path):
    with open(landmark_path+i,'r+') as f:
        one_landmark_file=np.asarray(json.load(f))[:,33:106,:]
        landmarks_by_file[i[0:-5]+'/']=one_landmark_file
label={}
annocation_v=os.listdir(label_path)
for i in annocation_v:
    with open(label_path+i,'r+') as f:
        temp_fx=f.read().split('\n')[1:-1]
        temp_fx_=[i.split(',') for i in temp_fx]
        temp_fx_ = np.asarray(temp_fx_)
        temp_fx_=temp_fx_.astype(np.int)
    label[i[:-4]+'/']=temp_fx_
print("*"*10+'data already'+"*"*10)


def mask(x):
    if x==-1:
        return 0
    if x==0:
        # return 0.25
        return 0.25
    if x==1:
        return 1
def re_target(x):
    if x==-1:
        return 0
    else:
        return x

total_P=np.zeros((8))#总共预测的P
TP = np.zeros((8))#预测的P中正确的个数
one_TP=np.zeros((8))
total_D=np.zeros((8))#数据中总共有的P
T=0
acc=0

videos=[v[:-4]+'/' for v in os.listdir(label_path)]
print('validing')
for v in videos:
    frames=os.listdir(seg_path+v)
    # print(frames)
    for f in range(len(frames)-2):
        print(v,f)
        T+=1

        # tt_data_prio = np.asarray(Image.open(seg_path +v+frames[f] ), dtype=np.uint8) / 256.0
        # tt_data_c = np.asarray(Image.open(seg_path +v+frames[f+1] ), dtype=np.uint8) / 256.0
        # # cv2.imshow('fff',Image.open(seg_path +v+frames[f+1]))
        # tt_data_c_boder = np.asarray(Image.open(boder_path +v+frames[f+2] ), dtype=np.uint8) / 256.0
        # tt_data_post = np.asarray(Image.open(seg_path +v+frames[f]), dtype=np.uint8) / 256.0
        # tt_landmark_prio = np.reshape(landmarks_by_file[v][f], (landmarks_by_file[v][f].shape[0] * landmarks_by_file[v][f].shape[1],))
        # tt_landmark_c = np.reshape(landmarks_by_file[v][f+1], (landmarks_by_file[v][f].shape[0] * landmarks_by_file[v][f].shape[1],))
        # tt_landmark_post = np.reshape(landmarks_by_file[v][f+2], (landmarks_by_file[v][f].shape[0] * landmarks_by_file[v][f].shape[1],))
        # landmark = np.concatenate((tt_landmark_post - tt_landmark_prio, tt_landmark_c), axis=-1)
        # cur_label=np.asarray(label[v][f+1])


        # tt_data_prio = np.asarray(Image.open(seg_path +v+frames[f] ), dtype=np.uint8) / 256.0
        tt_data_c = np.asarray(Image.open(seg_path +v+frames[f+1] ), dtype=np.uint8) / 256.0
        # # cv2.imshow('fff',Image.open(seg_path +v+frames[f+1]))
        tt_data_c_boder = np.asarray(Image.open(boder_path +v+frames[f+2] ), dtype=np.uint8) / 256.0

        tt_landmark_prio = np.reshape(landmarks_by_file[v][f], (landmarks_by_file[v][f].shape[0] * landmarks_by_file[v][f].shape[1],))
        tt_landmark_c=np.reshape(landmarks_by_file[v][f], (landmarks_by_file[v][f+1].shape[0] * landmarks_by_file[v][f].shape[1]))
        tt_landmark_post = np.reshape(landmarks_by_file[v][f+2], (landmarks_by_file[v][f].shape[0] * landmarks_by_file[v][f].shape[1],))
        # landmark = np.concatenate((tt_landmark_post - tt_landmark_prio, tt_landmark_c), axis=-1)
        landmark = tt_landmark_post - tt_landmark_prio

        cur_label=np.asarray(label[v][f+1])
        # landmark = tt_landmark_post - tt_landmark_prio

        images = torch.FloatTensor([np.concatenate([[tt_data_c],[tt_data_c_boder]], axis=0)]).cuda()
        seq_label = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7], ] * 1).cuda()
        landmark = torch.FloatTensor([landmark]).cuda()


        cnn_feature = cc(images)
        landmark_feature = ss(landmark)
        out = aq(cnn_feature, landmark_feature)
        x = np.reshape(torch.argmax(out, dim=1).detach().cpu().numpy(), (1, 8))

        one_TP += x[0]* cur_label
        total_P += x[0]
        total_D += np.reshape(cur_label, (8))
        acc += np.sum(x == cur_label,)
total_P += np.asarray([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,])
total_D += np.asarray([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,])
print(total_D)
Precise = one_TP / total_P
Recall = one_TP / total_D
F1 = np.mean(2 * (Precise * Recall) / (Precise + Recall + 0.0001))
acc = acc/ (T * 8 * 1.0)

print('Percormance:', 0.5 * acc + 0.5 * F1, 'ACC:', acc, 'F1:', F1)
print('Precise:', list(Precise))
print('Recall：', list(Recall))

