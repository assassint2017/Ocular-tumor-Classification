"""

目前疾病分类的做法是
根据医生给出的金标准，在病灶最明显的一张slice上选取感兴趣区域，仅对此区域提取特征
将T1C和T2两个模态的数据并在一起使用
"""

import os
import shutil

import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F


data_dirs = [
    '/home/zcy/Desktop/Ocular-tumor/raw_data/lbl/',
    '/home/zcy/Desktop/Ocular-tumor/raw_data/yz/'
]

new_data_dir = '/home/zcy/Desktop/Ocular-tumor/data/'
new_lbl_dir = '/home/zcy/Desktop/Ocular-tumor/data/lbl/'
new_yz_dir = '/home/zcy/Desktop/Ocular-tumor/data/yz/'

if os.path.exists(new_data_dir):
    shutil.rmtree(new_data_dir)

os.mkdir(new_data_dir)
os.mkdir(new_lbl_dir)
os.mkdir(new_yz_dir)


def scale(array):
    """

    将矩阵的维度统一
    """
    array = torch.FloatTensor(array).unsqueeze(dim=0).unsqueeze(dim=0)
    array = F.upsample(array, (512, 512), mode='bilinear').squeeze(0).squeeze(0).numpy()
    array = np.round(array).astype(np.int16)

    return array


def get_minimal_square(array):
    """

    获得病变部位的最小包围框,然后对包围框进行padding到224*224分辨率
    """
    x = np.any(array, axis=0)
    start_x, end_x = np.where(x)[0][[0, -1]]

    y = np.any(array, axis=1)
    start_y, end_y = np.where(y)[0][[0, -1]]

    temp_array = array[start_y:end_y + 1, start_x:end_x + 1]

    x_before = int((224 - (end_x - start_x + 1)) / 2)
    x_after = 224 - x_before - (end_x - start_x + 1)
    y_before = int((224 - (end_y - start_y + 1)) / 2)
    y_after = 224 - y_before - (end_y - start_y + 1)

    array = np.pad(temp_array, ((y_before, y_after), (x_before, x_after)), 'constant', constant_values=0)
    return array


for data_dir_index, data_dir in enumerate(data_dirs):
    for patient_dir in os.listdir(data_dir):

        # 读取数据
        T1C = sitk.ReadImage(os.path.join(data_dir, patient_dir, 'T1C.mha'), sitk.sitkInt16)
        T1C_array = sitk.GetArrayFromImage(T1C)

        T2 = sitk.ReadImage(os.path.join(data_dir, patient_dir, 'T2.mha'), sitk.sitkInt16)
        T2_array = sitk.GetArrayFromImage(T2)

        T1C_mask = sitk.ReadImage(os.path.join(data_dir, patient_dir, 'mask_C.mha'), sitk.sitkUInt8)
        T1C_mask = sitk.GetArrayFromImage(T1C_mask)

        T2_mask = sitk.ReadImage(os.path.join(data_dir, patient_dir, 'mask_T2.mha'), sitk.sitkUInt8)
        T2_mask = sitk.GetArrayFromImage(T2_mask)

        # 拼接数据
        print('-------------------------------')
        T1C_slice_index = np.where(np.any(T1C_mask, axis=(1, 2)))
        T2_slice_index = np.where(np.any(T2_mask, axis=(1, 2)))

        T1C_array = np.squeeze(T1C_array[T1C_slice_index])
        T1C_mask = np.squeeze(T1C_mask[T1C_slice_index])

        T2_array = np.squeeze(T2_array[T2_slice_index])
        T2_mask = np.squeeze(T2_mask[T2_slice_index])

        # 金标准的分辨率和数据有可能不一样，所以首先应该进行缩放
        if T1C_array.shape[0] != 512 or T1C_array.shape[1] != 512:
            # 缩放到指定的大小
            T1C_array = scale(T1C_array)

        if T2_array.shape[0] != 512 or T2_array.shape[1] != 512:
            # 缩放到指定的大小
            T2_array = scale(T2_array)

        if T2_mask.shape[0] != 512 or T2_mask.shape[1] != 512:
            # 缩放到指定的大小
            T2_mask = scale(T2_mask)

        if T1C_mask.shape[0] != 512 or T1C_mask.shape[1] != 512:
            # 缩放到指定的大小
            T1C_mask = scale(T1C_mask)

        T1C_array = T1C_array * T1C_mask
        T2_array = T2_array * T2_mask

        print('T1C_array shape:', T1C_array.shape)
        print('T2_array shape:', T2_array.shape)

        T1C_array = get_minimal_square(T1C_array)
        T2_array = get_minimal_square(T2_array)

        print('T1C_array shape:', T1C_array.shape)
        print('T2_array shape:', T2_array.shape)

        new_mr_array = np.stack((T1C_array, T2_array), axis=0)
        print('new data shape:', new_mr_array.shape)
        if new_mr_array.shape[0] != 2 or new_mr_array.shape[1] != 224 or new_mr_array.shape[2] != 224:
            print('error')

        # 保存数据
        new_mr = sitk.GetImageFromArray(new_mr_array)

        new_mr.SetDirection(T1C.GetDirection())
        new_mr.SetOrigin(T1C.GetOrigin())
        new_mr.SetSpacing(T1C.GetSpacing())

        new_mr_name = patient_dir + '.mha'

        if data_dir_index is 0:
            sitk.WriteImage(new_mr, os.path.join(new_lbl_dir, new_mr_name))
        else:
            sitk.WriteImage(new_mr, os.path.join(new_yz_dir, new_mr_name))

# 最缩放到512*512分辨率后,由于测试过所有的病变部位的大小最大不超过200*200
# 所以最终的包围框的大小就选为了224*224,经典的图像分类的大小
