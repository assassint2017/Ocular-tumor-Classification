"""

病变部位分析脚本
"""


import os
import numpy as np
import SimpleITK as sitk


yz_dir = '/home/zcy/Desktop/Ocular-tumor/raw_data/yz/'
lbl_dir = '/home/zcy/Desktop/Ocular-tumor/raw_data/lbl/'


z_list = []
y_list = []
x_list = []

for item in os.listdir(yz_dir):

    try:
        T1C = sitk.ReadImage(os.path.join(yz_dir, item, 'T1C.mha'))
        T1C = sitk.GetArrayFromImage(T1C)

        lesion = sitk.ReadImage(os.path.join(yz_dir, item, 'mask_C.mha'))
        lesion = sitk.GetArrayFromImage(lesion)

    except Exception as e:
        print(item)
        print('error')
        print('------------------------')
        continue

    print(item)
    print(T1C.shape)
    print(lesion.shape)

    if T1C.shape[1:] != lesion.shape[1:]:
        print('Axial not match!')
    if T1C.shape[0] != lesion.shape[0]:
        print('Longitudinal not match!')

    z = np.any(lesion, axis=(1, 2))
    start_z, end_z = np.where(z)[0][[0, -1]]

    x = np.any(lesion, axis=(0, 1))
    start_x, end_x = np.where(x)[0][[0, -1]]

    y = np.any(lesion, axis=(0, 2))
    start_y, end_y = np.where(y)[0][[0, -1]]

    print('z:', end_z - start_z + 1)
    print('y:', end_y - start_y + 1)
    print('x:', end_x - start_x + 1)

    z_list.append(end_z - start_z + 1)
    y_list.append(end_y - start_y + 1)
    x_list.append(end_x - start_x + 1)
    print('------------------------')


print('++++++++++++++++')

for item in os.listdir(lbl_dir):

    try:
        T1C = sitk.ReadImage(os.path.join(lbl_dir, item, 'T1C.mha'))
        T1C = sitk.GetArrayFromImage(T1C)

        lesion = sitk.ReadImage(os.path.join(lbl_dir, item, 'mask_C.mha'))
        lesion = sitk.GetArrayFromImage(lesion)

    except Exception as e:
        print(item)
        print('error')
        print('------------------------')
        continue

    print(item)
    print(T1C.shape)
    print(lesion.shape)
    if T1C.shape[1:] != lesion.shape[1:]:
        print('Axial not match!')
    if T1C.shape[0] != lesion.shape[0]:
        print('Longitudinal not match!')

    z = np.any(lesion, axis=(1, 2))
    start_z, end_z = np.where(z)[0][[0, -1]]

    x = np.any(lesion, axis=(0, 1))
    start_x, end_x = np.where(x)[0][[0, -1]]

    y = np.any(lesion, axis=(0, 2))
    start_y, end_y = np.where(y)[0][[0, -1]]

    print('z:', end_z - start_z + 1)
    print('y:', end_y - start_y + 1)
    print('x:', end_x - start_x + 1)

    z_list.append(end_z - start_z + 1)
    y_list.append(end_y - start_y + 1)
    x_list.append(end_x - start_x + 1)
    print('------------------------')

print(sum(z_list) / len(z_list))
print(sum(y_list) / len(y_list))
print(sum(x_list) / len(x_list))
print('-------------------------')
print(max(z_list), min(z_list))
print(max(y_list), min(y_list))
print(max(x_list), min(x_list))


# 发现mask_C分割金标准和原始T1C数据的维度有一些是不一致的(轴向和纵向都会出现不一致),轴向13例,纵向6例
# 纵向一般缺少一张或者两张slcie,轴向是514*512 512*512
# 所有数据读取是没有问题的
# 下面是病灶平均大小
# 98.35714285714286
# 78.92857142857143
