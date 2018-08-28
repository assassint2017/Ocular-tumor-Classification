"""

对已经做好的cv数据集进行分析
"""

import os

import SimpleITK as sitk

cv_data_dir = '/home/zcy/Desktop/Ocular-tumor/cv_data/'

lbl_list = []
yz_list = []
mri_list = []

for fold in os.listdir(cv_data_dir):
    lbl_list.append(os.path.join(cv_data_dir, fold, 'lbl'))
    yz_list.append(os.path.join(cv_data_dir, fold, 'yz'))

for lbl_dir, yz_dir in zip(lbl_list, yz_list):
    temp_lbl_list = os.listdir(lbl_dir)
    temp_yz_list = os.listdir(yz_dir)

    lbl_list = list(map(lambda x: os.path.join(lbl_dir, x), temp_lbl_list))
    yz_list = list(map(lambda x: os.path.join(yz_dir, x), temp_yz_list))

    mri_list += (lbl_list + yz_list)

for mri_dir in mri_list:

    mri = sitk.ReadImage(mri_dir)
    mri_array = sitk.GetArrayFromImage(mri)

    print(mri_array.max(), mri_array.mean())

