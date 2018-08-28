"""

数据分析脚本
"""

import os
import SimpleITK as sitk


yz_dir = '/home/zcy/Desktop/Ocular-tumor/raw_data/yz/'
lbl_dir = '/home/zcy/Desktop/Ocular-tumor/raw_data/lbl/'

min_slice = 100000
max_slice = 0

for item in os.listdir(yz_dir):
    T1C = sitk.ReadImage(os.path.join(yz_dir, item, 'T1C.mha'))
    T2 = sitk.ReadImage(os.path.join(yz_dir, item, 'T2.mha'))

    T1C = sitk.GetArrayFromImage(T1C)
    T2 = sitk.GetArrayFromImage(T2)

    if T1C.shape[1:] != T2.shape[1:]:
        print('Axial not match!')
    if T1C.shape[0] != T2.shape[0]:
        print('Longitudinal not match!')
    if T1C.shape[1] != 512 or T1C.shape[2] != 512 or T2.shape[1] != 512 or T2.shape[2] != 512:
        print('not 512')

    if T1C.shape[0] > max_slice:
        max_slice = T1C.shape[0]
    if T2.shape[0] > max_slice:
        max_slice = T2.shape[0]
    if T1C.shape[0] < min_slice and T1C.shape[0] is not 1:
        min_slice = T1C.shape[0]
    if T2.shape[0] < min_slice and T2.shape[0] is not 1:
        min_slice = T2.shape[0]

    print(T1C.shape)
    print(T1C.max(), T1C.min())
    print(T2.shape)
    print(T2.max(), T2.min())
    print(item)
    print('--------------------')


print('----------')

for item in os.listdir(lbl_dir):
    T1C = sitk.ReadImage(os.path.join(lbl_dir, item, 'T1C.mha'))
    T2 = sitk.ReadImage(os.path.join(lbl_dir, item, 'T2.mha'))

    T1C = sitk.GetArrayFromImage(T1C)
    T2 = sitk.GetArrayFromImage(T2)

    if T1C.shape[1:] != T2.shape[1:]:
        print('Axial not match!')
    if T1C.shape[0] != T2.shape[0]:
        print('Longitudinal not match!')
    if T1C.shape[1] != 512 or T1C.shape[2] != 512 or T2.shape[1] != 512 or T2.shape[2] != 512:
        print('not 512')

    if T1C.shape[0] > max_slice:
        max_slice = T1C.shape[0]
    if T2.shape[0] > max_slice:
        max_slice = T2.shape[0]
    if T1C.shape[0] < min_slice and T1C.shape[0] is not 1:
        min_slice = T1C.shape[0]
    if T2.shape[0] < min_slice and T2.shape[0] is not 1:
        min_slice = T2.shape[0]

    print(T1C.shape)
    print(T1C.max(), T1C.min())
    print(T2.shape)
    print(T2.max(), T2.min())
    print(item)
    print('--------------------')

print(max_slice)
print(min_slice)

# lbl 84例 yz 73例，共157例数据
# MRI数据的灰度值基本都是从0开始的,但是有一些有例外数据的最小灰度值为-32768.最高值有六七百的,也有三四千的
# 眼部的这批数据轴向分辨率几乎都是512*512但是也有少数例外30例，但是这样的数据直接缩放到512*512是没有问题的
# T1C和T2两种模态，绝大部分都是一样的分辨率，但是也要极少数的例外 20例,其中17例都是横断面分辨率不一致,3例数据slice数量不一致
# slice数量范围:10--16，15是数量最多的，能占三分之二
# lbl patient041的T2有问题，只有一张slice,对照分割标注查看了一下,发现是数据缺失了一部分,这例数据直接删除掉好了
# lbl patient081的T1C和T2同样是数据缺失问题
# yz patient055的T1C同样是数据缺失
# 对于以上的三个数据,直接删除掉好了
# 最终lbl 82例 yz 72例，共154例数据
