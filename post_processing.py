import numpy as np
import os

durable=3

label_root_path='./prediction/'
labels_files_name=os.listdir(label_root_path)

def post_processing(array):
    fade=1/durable
    temp_act=0
    temp_array=[]
    for i in array:
        if i >0:
            temp_act=1
            temp_array.append(1)
        else:
            temp_act-=fade
            if temp_act>0:
                temp_array.append(1)
            else:
                temp_array.append(0)
    return np.asarray(temp_array)
for i in labels_files_name:
    array_label=np.loadtxt(label_root_path+i, dtype=np.int64, delimiter=',', converters=None, skiprows=1, usecols=None, unpack=False, ndmin=0)
    temp=[]
    for c in range(8):
        array_label_colum=array_label[:,c]
        x=post_processing(array_label_colum)
        temp.append(x)
    post_label=np.vstack(temp)
    post_label=np.transpose(post_label,(1,0))
    print(post_label.shape)
    np.savetxt('./post_prediction/' +i, post_label, fmt="%d", delimiter=',',
               header='AU1,AU2,AU4,AU6,AU12,AU15,AU20,AU25', comments="")




