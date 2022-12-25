# -*- coding: utf-8 -*-
# @Time    : 25/12/2022 4:41 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
import glob
import shutil
import os

img_path = glob.glob('/media/breeze/dev/Mf_Cls/data/T2W_images/*')
label_path = glob.glob('/media/breeze/dev/Mf_Cls/data/T2W_labels/*')
cnt = 0
for i in img_path:
    cnt += 1
    if cnt % 5 == 0:
        shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/test/T2W_images')
    else:
        shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/train/T2W_images')
cnt = 0
for i in label_path:
    cnt += 1
    if cnt % 5 == 0:
        shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/test/T2W_labels')
    else:
        shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/train/T2W_labels')
