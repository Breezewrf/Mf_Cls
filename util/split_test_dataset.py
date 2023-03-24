# -*- coding: utf-8 -*-
# @Time    : 25/12/2022 4:41 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
import glob
import shutil
import os

adc_img_path = sorted(glob.glob('/media/breeze/dev/Mf_Cls/data/ADC_images/*'))
t2w_img_path = sorted(glob.glob('/media/breeze/dev/Mf_Cls/data/T2W_images/*'))
label_path = sorted(glob.glob('/media/breeze/dev/Mf_Cls/data/T2W_labels/*'))
cnt = 0
print(adc_img_path)
print(t2w_img_path)


for i in adc_img_path:
    cnt += 1
    if cnt % 5 == 0:
        shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/test/ADC_images/')
        print('adc: ', i)
    else:
        shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/train/ADC_images/')
cnt = 0
for i in t2w_img_path:
    cnt += 1
    if cnt % 5 == 0:
        shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/test/T2W_images/')
        print('t2w: ', i)
    else:
        shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/train/T2W_images/')
cnt = 0
for i in label_path:
    cnt += 1
    if cnt % 5 == 0:
        shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/test/labels/')
    else:
        shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/train/labels/')
