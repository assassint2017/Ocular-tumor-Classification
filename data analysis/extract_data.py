"""

从原始dcm数据中提取mha数据
此脚本暂时用不上了
"""

import os
import shutil

import SimpleITK as sitk

data_dirs = [
    '/home/zcy/Desktop/eye/lbl/',
    '/home/zcy/Desktop/eye/yz/'
]

for data_dir in data_dirs:
    for index, patient_dir in enumerate(os.listdir(data_dir)):

        # 提取T1C序列数据
        fullpath = os.path.join(data_dir, patient_dir, 'T1C/')
        fileid = sitk.ImageSeriesReader.GetGDCMSeriesIDs(fullpath)
        filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(fullpath, fileid[0])
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(filenames)
        mha_data = series_reader.Execute()
        sitk.WriteImage(mha_data, os.path.join(data_dir, patient_dir, 'T1C_data.mha'))

        # 提取T2序列数据
        fullpath = os.path.join(data_dir, patient_dir, 'T2/')
        fileid = sitk.ImageSeriesReader.GetGDCMSeriesIDs(fullpath)
        filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(fullpath, fileid[0])
        series_reader.SetFileNames(filenames)
        mha_data = series_reader.Execute()
        sitk.WriteImage(mha_data, os.path.join(data_dir, patient_dir, 'T2_data.mha'))

        # 移动mask数据
        os.system('mv ' + os.path.join(data_dir, patient_dir, 'mask/* ') + os.path.join(data_dir, patient_dir))

        # 删除多余的数据
        shutil.rmtree(os.path.join(data_dir, patient_dir, 'T1C/'))
        shutil.rmtree(os.path.join(data_dir, patient_dir, 'T2/'))
        shutil.rmtree(os.path.join(data_dir, patient_dir, 'mask/'))

        try:
            # 有些数据是缺少T1序列的
            shutil.rmtree(os.path.join(data_dir, patient_dir, 'T1/'))
        except Exception as e:
            print('missing T1 data')

        # 处理完一个数据打印一次序号
        print(index)
