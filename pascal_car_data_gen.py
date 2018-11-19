# -*- coding: utf-8 -*-
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd


def load_mats(dir_path, joint_names):
    files = os.listdir(dir_path)
    files_mat = [i for i in files if i.endswith('.mat')]

    result_mat = []

    for mat_file_name in files_mat:
        mat = loadmat(os.path.join(dir_path, mat_file_name))
        bbox = mat['record'][0, 0]['objects']['bbox'][0, 0]
        anchors = mat['record'][0, 0]['objects']['anchors'][0, 0][0, 0]
        if joint_names is None:
            joint_names = mat['record']['objects'][0,
                                                   0]['anchors'][0][0].dtype.names

        result = mat['record']['filename'][0, 0].reshape([1, -1])
        result = np.concatenate([result, bbox], axis=1)
        for anchor in anchors:
            status = anchor[0, 0]['status'][0, 0]
            loc = anchor[0, 0]['location']
            if status != 1:
                loc = np.array([-1, -1]).reshape([1, -1])

            result = np.concatenate([result, loc], axis=1)

        result_mat.append(result)

    result_mat = np.array(result_mat).squeeze(axis=1)
    return result_mat, joint_names


return_mat, joint_names = load_mats(
    '/home/vinnamkim/PASCAL3D+_release1.1/Annotations/car_imagenet', None)

df = pd.DataFrame(return_mat)
df.to_csv('temp.txt', sep=" ", header=None, index=False)
print('end')
