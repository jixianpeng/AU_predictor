
'''coding=utf-8'''
import os
import cv2
import numpy as np
from PIL import Image
import json
import torchvision
root_path='./cropped_aligned/'
root_path_s='./train/seg/'
root_path_boder='./train/boder/'
train_path='./Training_Set/'
valid_path='./Validation_Set/'
file_list=[]
dir_list=[]
annocation_t=os.listdir(train_path)
annocation_v=os.listdir(valid_path)
train_label={}
valid_label={}
train_data={}
valid_data={}
print('loading data')
for i in annocation_t:
    with open(train_path+i,'r+') as f:
        temp_fx=f.read().split('\n')[1:-1]
        temp_fx_=[i.split(',') for i in temp_fx]
        temp_fx_ = np.asarray(temp_fx_)
        temp_fx_=temp_fx_.astype(np.int)
    train_label[i[:-4]]=temp_fx_
for i in annocation_v:
    with open(valid_path+i,'r+') as f:
        temp_fx=f.read().split('\n')[1:-1]
        temp_fx_=[i.split(',') for i in temp_fx]
        temp_fx_ = np.asarray(temp_fx_)
        temp_fx_=temp_fx_.astype(np.int)
    valid_label[i[:-4]]=temp_fx_
for i in annocation_t:
    temp_image=os.listdir(root_path+i[:-4])
    train_data[i[:-4]]=[im for im in temp_image if im.endswith('jpg')]
for i in annocation_v:
    temp_image=os.listdir(root_path+i[:-4])
    valid_data[i[:-4]]=[im for im in temp_image if im.endswith('jpg')]
train_label_keys=np.asarray(list(train_label.keys()))
train_label_keys_len=len(train_label_keys)
print('data already')
def load_json_landmark():
    landmark_dataset={}
    for i in os.listdir('./train/' + 'landmark/'):
        if i.endswith('.json'):
            print(i)
            temp = json.load(open('./train/' + 'landmark/' + i, 'r+'))
            temp = np.asarray(temp)[:,33:106,:]
            landmark_dataset[i[0:-5]]=temp
    return landmark_dataset
print('loading landmark')
landmark_dataset=load_json_landmark()
print('landmark already')



# def get_batch(batch_size):
#     v_index=np.random.randint(0,train_label_keys_len,batch_size)
#     v_name=train_label_keys[v_index]
#     temp_data=[]
#     temp_label=[]
#     temp_landmark=[]
#     for i in v_name:
#         num_frame=len(train_data[i])
#         while True:
#             frame_selected_index = np.random.randint(0, num_frame)
#             try:
#
#                 image_selected_c = train_data[i][frame_selected_index]
#
#                 tt_data_c=np.asarray(Image.open(root_path_s + i + '/' + image_selected_c), dtype=np.uint8)/256.0
#                 tt_data_c_boder = np.asarray(Image.open(root_path_boder + i + '/' + image_selected_c),dtype=np.uint8) / 256.0
#
#                 images=np.concatenate([[tt_data_c],[tt_data_c_boder]],axis=0)
#
#                 tt_landmark_prio=np.reshape(landmark_dataset[i][frame_selected_index - 2],
#                                             (landmark_dataset[i][frame_selected_index - 2].shape[0]*
#                                              landmark_dataset[i][frame_selected_index - 2].shape[1],))
#                 # tt_landmark_c=np.reshape(landmark_dataset[i][frame_selected_index ],
#                 #                             (landmark_dataset[i][frame_selected_index ].shape[0]*
#                 #                              landmark_dataset[i][frame_selected_index].shape[1],))
#                 tt_landmark_post = np.reshape(landmark_dataset[i][frame_selected_index + 2],
#                                               (landmark_dataset[i][frame_selected_index + 2].shape[0]*
#                                              landmark_dataset[i][frame_selected_index +2].shape[1],))
#                 tt_landmark=tt_landmark_post-tt_landmark_prio
#                 break
#             except:
#                 continue
#         temp_data.append(images)
#         one_label=train_label[i][int(image_selected_c[0:-4])-1]
#         #如果不是array。那么在采用np.where的时候虽然不报错但是并不能真正按要求实现功能
#         one_label=np.asarray(one_label)
#         one_label = np.where(one_label == 1)[0]
#         if len(one_label)!=0:
#             one_label+=1
#             one_label = np.pad(one_label, ((0, 9 - len(one_label))), 'constant', constant_values=(0, -1))
#         else:
#             one_label=np.asarray([0])
#             one_label = np.pad(one_label, ((0, 9 - len(one_label))), 'constant', constant_values=(0, -1))
#         temp_label.append(one_label)
#         temp_landmark.append(tt_landmark)
#     temp_data=np.asarray(temp_data)
#     temp_label=np.asarray(temp_label)
#     temp_landmark=np.asarray(temp_landmark)
#     return temp_data,temp_label,temp_landmark
def get_batch(batch_size):
    v_index=np.random.randint(0,train_label_keys_len,batch_size)
    v_name=train_label_keys[v_index]
    temp_data=[]
    temp_label=[]
    temp_landmark=[]
    for i in v_name:
        num_frame=len(train_data[i])
        while True:
            frame_selected_index = np.random.randint(0, num_frame)
            try:

                image_selected_c = train_data[i][frame_selected_index]

                tt_data_c=np.asarray(Image.open(root_path_s + i + '/' + image_selected_c), dtype=np.uint8)/256.0
                tt_data_c_boder = np.asarray(Image.open(root_path_boder + i + '/' + image_selected_c),dtype=np.uint8) / 256.0

                images=np.concatenate([[tt_data_c],[tt_data_c_boder]],axis=0)

                tt_landmark_prio=np.reshape(landmark_dataset[i][frame_selected_index - 2],
                                            (landmark_dataset[i][frame_selected_index - 2].shape[0]*
                                             landmark_dataset[i][frame_selected_index - 2].shape[1],))
                tt_landmark_post = np.reshape(landmark_dataset[i][frame_selected_index + 2],
                                              (landmark_dataset[i][frame_selected_index + 2].shape[0]*
                                             landmark_dataset[i][frame_selected_index +2].shape[1],))
                tt_landmark=tt_landmark_post-tt_landmark_prio
                break
            except:
                continue
        temp_data.append(images)
        one_label=train_label[i][int(image_selected_c[0:-4])-1]
        temp_label.append(one_label)
        temp_landmark.append(tt_landmark)
    temp_data=np.asarray(temp_data)
    temp_label=np.asarray(temp_label)
    temp_label = np.reshape(np.asarray(temp_label), (batch_size * 8,))
    temp_landmark=np.asarray(temp_landmark)

    return temp_data,temp_label,temp_landmark



