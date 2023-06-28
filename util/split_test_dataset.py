# -*- coding: utf-8 -*-
# @Time    : 25/12/2022 4:41 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
import glob
import shutil
import os

"""
Create new folder and split the train set and test set
"""
dataset_name = 'PWH'
dwi_img_path = sorted(glob.glob('/media/breeze/dev/Mf_Cls/data/{}/DWI_images/*'.format(dataset_name)))
adc_img_path = sorted(glob.glob('/media/breeze/dev/Mf_Cls/data/{}/ADC_images/*'.format(dataset_name)))
t2w_img_path = sorted(glob.glob('/media/breeze/dev/Mf_Cls/data/{}/T2W_images/*'.format(dataset_name)))
label_path = sorted(glob.glob('/media/breeze/dev/Mf_Cls/data/{}/segmentation/*'.format(dataset_name)))

print(adc_img_path)
print(t2w_img_path)
os.makedirs(os.path.dirname('/media/breeze/dev/Mf_Cls/data/{}/test/ADC_images/'.format(dataset_name)), exist_ok=True)
os.makedirs(os.path.dirname('/media/breeze/dev/Mf_Cls/data/{}/test/DWI_images/'.format(dataset_name)), exist_ok=True)
os.makedirs(os.path.dirname('/media/breeze/dev/Mf_Cls/data/{}/test/T2W_images/'.format(dataset_name)), exist_ok=True)
os.makedirs(os.path.dirname('/media/breeze/dev/Mf_Cls/data/{}/test/labeled_GT_colored/'.format(dataset_name)), exist_ok=True)
os.makedirs(os.path.dirname('/media/breeze/dev/Mf_Cls/data/{}/train/ADC_images/'.format(dataset_name)), exist_ok=True)
os.makedirs(os.path.dirname('/media/breeze/dev/Mf_Cls/data/{}/train/DWI_images/'.format(dataset_name)), exist_ok=True)
os.makedirs(os.path.dirname('/media/breeze/dev/Mf_Cls/data/{}/train/T2W_images/'.format(dataset_name)), exist_ok=True)
os.makedirs(os.path.dirname('/media/breeze/dev/Mf_Cls/data/{}/train/labeled_GT_colored/'.format(dataset_name)), exist_ok=True)

"""
for ADC images
"""
# cnt = 0
# record = []
# for i in adc_img_path:
#     lesion_id = i.split('/')[-1].split('_')[1]
#     if lesion_id not in record:
#         record.append(lesion_id)
#         cnt += 1
#     if cnt % 5 == 0:
#         shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/{}/test/ADC_images/'.format(dataset_name))
#         print('adc: ', i)
#     else:
#         shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/{}/train/ADC_images/'.format(dataset_name))
#
# """
# for T2W images
# """
# cnt = 0
# record = []
# for i in t2w_img_path:
#     lesion_id = i.split('/')[-1].split('_')[1]
#     if lesion_id not in record:
#         record.append(lesion_id)
#         cnt += 1
#     if cnt % 5 == 0:
#         shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/{}/test/T2W_images/'.format(dataset_name))
#         print('t2w: ', i)
#     else:
#         shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/{}/train/T2W_images/'.format(dataset_name))
#
# """
# for GT images
# """
# cnt = 0
# record = []
# for i in label_path:
#     lesion_id = i.split('/')[-1].split('_')[1]
#     if lesion_id not in record:
#         record.append(lesion_id)
#         cnt += 1
#     if cnt % 5 == 0:
#         shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/{}/test/labeled_GT_colored/'.format(dataset_name))
#         print('labels: ', i)
#     else:
#         shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/{}/train/labeled_GT_colored/'.format(dataset_name))
#
# """
# for DWI images
# """
# cnt = 0
# record = []
# for i in dwi_img_path:
#     lesion_id = i.split('/')[-1].split('_')[1]
#     if lesion_id not in record:
#         record.append(lesion_id)
#         cnt += 1
#     if cnt % 5 == 0:
#         shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/{}/test/DWI_images/'.format(dataset_name))
#         print('dwi: ', i)
#     else:
#         shutil.copy(i, '/media/breeze/dev/Mf_Cls/data/{}/train/DWI_images/'.format(dataset_name))

"""
reformat the PWH data to ProstateX form
"""
# import os
#
# # 指定文件夹路径
# folder_path = "/media/breeze/dev/Mf_Cls/data/PWH/test/T2W_images"
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     # 仅处理以 ".jpg" 结尾的文件
#     if filename.endswith(".jpg"):
#         # 解析文件名中的数字
#         parts = filename.split("_")
#         num1 = parts[1]
#         num2 = parts[2].split(".")[0]
#         # 构造新的文件名
#         new_filename = f"ProstateX-{num1}-{num2}.jpg"
#         # 修改文件名
#         os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

"""
Add classification label to PWH dataset
"""
# import os
# from glob import glob
# path = '/media/breeze/dev/Mf_Cls/data/PWH/cls_label/oneLesion_label.txt'
# with open(path) as f:
#     lines = f.readlines()
#     for line in lines:
#         """
#         each line denotes a patient,
#         slices include the MRI slices of the patient
#         """
#         patient_id = int(float(line.split(" ")[0]))
#         print(patient_id)
#         slices = glob('/media/breeze/dev/Mf_Cls/data/PWH/test/labeled_GT_colored/ProstateX-' + str(patient_id) + '-*')
#         print(slices)
#         grades = []
#         for i in line.split(" ")[1:]:
#             grades.append(int(float(i)))
#         print(grades)
#         for s in slices:
#             splited = s.split('.')
#             splited[0] += ('-' + str(max(grades)+1))
#             new_file_name = '.'.join(splited)
#             print(new_file_name)
#             os.rename(s, new_file_name)
