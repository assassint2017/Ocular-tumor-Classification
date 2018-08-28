"""

约定:
lbl为1
yz为0
"""

import os
import random

import SimpleITK as sitk
from scipy import ndimage
import torch
from torch.utils.data import Dataset as dataset


class Dataset(dataset):
    def __init__(self, lbl_dirs, yz_dirs, istraining):

        self.istraining = istraining

        self.mri_list = []

        if isinstance(lbl_dirs, str) and isinstance(yz_dirs, str):
            lbl_dirs = [lbl_dirs]
            yz_dirs = [yz_dirs]

        for lbl_dir, yz_dir in zip(lbl_dirs, yz_dirs):

            lbl_list = os.listdir(lbl_dir)
            yz_list = os.listdir(yz_dir)

            lbl_list = list(map(lambda x: os.path.join(lbl_dir, x), lbl_list))
            yz_list = list(map(lambda x: os.path.join(yz_dir, x), yz_list))

            self.mri_list += (lbl_list + yz_list)

    def __getitem__(self, index):

        mri_path = self.mri_list[index]

        # 将CT和金标准读入到内存中
        mri = sitk.ReadImage(mri_path, sitk.sitkInt16)
        mri_array = sitk.GetArrayFromImage(mri)

        if 'lbl' in mri_path:
            label = torch.LongTensor([1])
        else:
            label = torch.LongTensor([0])

        if self.istraining is True:
            # # 以0.5的概率进行随机移动
            # if random.uniform(0, 1) >= 0.5:
            #     shift_x = random.uniform(-10, 10)
            #     shift_y = random.uniform(-10, 10)
            #     mri_array = ndimage.shift(mri_array, (0, shift_x, shift_y), order=0, mode='constant')
            #
            # # 以0.5的概率在10度的范围内随机旋转
            # # 角度为负数是顺时针旋转，角度为正数是逆时针旋转
            # if random.uniform(0, 1) >= 0.5:
            #     angle = random.uniform(-10, 10)
            #     mri_array = ndimage.rotate(mri_array, angle, axes=(1, 2), reshape=False, cval=0)
            pass

        mri_array = torch.FloatTensor(mri_array)

        return mri_array, label

    def __len__(self):

        return len(self.mri_list)
