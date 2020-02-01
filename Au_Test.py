'''coding=utf-8'''
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
boder_path='./test/boder/'
landmark_path='./test/landmarks/'
seg_path='./test/seg/'

video_path='./test/video/'
import cv2





landmarks_by_file={}
for i in os.listdir(landmark_path):
    print(i)
    with open(landmark_path+i,'r+') as f:
        one_landmark_file=np.asarray(json.load(f))[:,33:106,:]
        landmarks_by_file[i[0:-5]+'/']=one_landmark_file

print("*"*10+'data already'+"*"*10)


#按照视频帧数

videos=[v+'/' for v in os.listdir(boder_path)]
print('testing')
for v in videos:
    print(v)
    # v='video28/'
    t = 0
    success=True
    print(video_path + v[0:-1] + '.mp4')
    vvv=cv2.VideoCapture(video_path + v[0:-1] + '.mp4')
    while success:
        success, frame = vvv.read()
        t += 1
        print(success)

    frames=os.listdir(seg_path+v)
    temp_total_p=np.zeros(shape=(t,8),dtype=np.int64)
    for f in range(len(frames)-4):
        print(v,f)
        tt_data_c = np.asarray(Image.open(seg_path +v+frames[f+2] ), dtype=np.uint8) / 256.0
        tt_data_c_boder = np.asarray(Image.open(boder_path +v+frames[f+2] ), dtype=np.uint8) / 256.0

        tt_landmark_prio = np.reshape(landmarks_by_file[v][f], (landmarks_by_file[v][f].shape[0] * landmarks_by_file[v][f].shape[1],))
        tt_landmark_post = np.reshape(landmarks_by_file[v][f+4], (landmarks_by_file[v][f].shape[0] * landmarks_by_file[v][f].shape[1],))
        landmark = tt_landmark_post - tt_landmark_prio

        images = torch.FloatTensor([np.concatenate([[tt_data_c],[tt_data_c_boder]], axis=0)]).cuda()
        seq_label = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7], ] * 1).cuda()
        landmark = torch.FloatTensor([landmark]).cuda()


        cnn_feature = cc(images)
        landmark_feature = ss(landmark)
        out = aq(cnn_feature, landmark_feature)
        p = np.reshape(torch.argmax(out, dim=1).detach().cpu().numpy(), (1, 8))
        temp_total_p[int(frames[f + 2][0:-4:])-1]=p

    np.savetxt('./prediction/'+v[0:-1]+'.txt',temp_total_p,fmt="%d",delimiter=',',header='AU1,AU2,AU4,AU6,AU12,AU15,AU20,AU25'+'\n',)





