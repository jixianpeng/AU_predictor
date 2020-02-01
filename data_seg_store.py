'''coding=utf-8'''
import os
import numpy as np
batch_size=16
from  face_seg import *
root_path='D:/DATASET/aff_wild2/test/cropped_aligned/'
path='D:/DATASET/aff_wild2/test/seg/'

video=[i for i in os.listdir(root_path)]
# print(video)
# for i in video:
#     os.makedirs(path+i+'/')
for v in video:
    print(v)
    for f in os.listdir(root_path+v):
        if f.endswith('.jpg'):
            print(f)
            file_list=[root_path+v+'/'+f]
            images=seg(file_list)
            Image.fromarray(images[0]).save(path+v+'/'+f)
        else:
            continue
'''coding=utf-8'''
import os
import numpy as np
batch_size=16
from  face_seg import *
root_path='D:/DATASET/aff_wild2/test/cropped_aligned/'
path_s='D:/DATASET/aff_wild2/test/seg/'
path_='D:/DATASET/aff_wild2/test/boder/'

video=[i for i in os.listdir(path_s)]
# print(video)
# for i in video:
#     os.makedirs(path_+i+'/')
for v in video:
    print(v)
    for f in os.listdir(path_s+v):
        print(f)
        if f.endswith('.jpg'):
            path=path_s+v+'/'+f
            img = Image.open(path)  # 读图片并转化为灰度图
            img_array = np.array(img)  # 转化为数组
            w, h = img_array.shape
            img_border = np.zeros((w - 1, h - 1))
            for x in range(1, w - 1):
                for y in range(1, h - 1):
                    Sx = img_array[x + 1][y - 1] + 2 * img_array[x + 1][y] + img_array[x + 1][y + 1] - \
                         img_array[x - 1][y - 1] - 2 * \
                         img_array[x - 1][y] - img_array[x - 1][y + 1]
                    Sy = img_array[x - 1][y + 1] + 2 * img_array[x][y + 1] + img_array[x + 1][y + 1] - \
                         img_array[x - 1][y - 1] - 2 * \
                         img_array[x][y - 1] - img_array[x + 1][y - 1]
                    img_border[x][y] = (Sx * Sx + Sy * Sy) ** 0.5
            # img2 = Image.fromarray(img_border + img_array[0:-1, 0:-1])
            img2 = Image.fromarray(img_border).resize(size=(112,112)).convert('L')
            img2.save(path_+v+'/'+f)
        else:
            continue
#
#
#
#
