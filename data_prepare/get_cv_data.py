"""

可以使用的数据有132例
将数据分成五折进行交叉验证
前四折中:lbl:14 yz:12
最后一折:lbl:14 yz:14
"""

import os
import shutil
import random

# 交叉验证数据集的地址
cv_data_dir = '/home/zcy/Desktop/Ocular-tumor/cv_data/'

if os.path.exists(cv_data_dir):
    shutil.rmtree(cv_data_dir)
os.makedirs(cv_data_dir)

# 存放原始数据的地址
yz_dir = '/home/zcy/Desktop/Ocular-tumor/data/yz/'
lbl_dir = '/home/zcy/Desktop/Ocular-tumor/data/lbl/'

for fold_index in range(5):

    new_dir = os.path.join(cv_data_dir, 'fold' + str(fold_index), 'lbl')
    os.makedirs(new_dir)

    if fold_index != 4:

        for i in range(14):
            shutil.move(os.path.join(lbl_dir, random.choice(os.listdir(lbl_dir))), new_dir)
    else:
        os.system('mv ' + lbl_dir + '* ' + new_dir)

    new_dir = os.path.join(cv_data_dir, 'fold' + str(fold_index), 'yz')
    os.makedirs(new_dir)

    if fold_index != 4:
        for i in range(12):
            shutil.move(os.path.join(yz_dir, random.choice(os.listdir(yz_dir))), new_dir)
    else:
        os.system('mv ' + yz_dir + '* ' + new_dir)

shutil.rmtree('/home/zcy/Desktop/Ocular-tumor/data/')
