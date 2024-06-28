# -*- coding: utf-8 -*-
"""
@ Description: 
-------------
Band selection network with Fullly Connected Nets (aka. MLP)
-------------
@ Time    : 2019/2/28 15:32
@ Author  : Yaoming Cai
@ FileName: BS_Net_FC.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import time

import numpy as np
import sys

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.append('/home/caiyaom/python_codes/')
from utility import eval_band_cv
from Preprocessing import Processor
from sklearn.preprocessing import minmax_scale


class BS_Net_FC:

    def __init__(self, lr, batch_size, epoch, n_selected_band):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.n_selected_band = n_selected_band



    def fit(self, X, img=None, gt=None):
        score_list = []
        n_sam, n_channel = X.shape
        x_new = img
        n_row, n_clm, n_band = x_new.shape
        img_ = minmax_scale(x_new.reshape((n_row * n_clm, n_band))).reshape((n_row, n_clm, n_band))
        p = Processor()
        img_correct, gt_correct = p.get_correct(img_, gt)
        score = eval_band_cv(img_correct, gt_correct, times=1, test_size=0.95)
        print('acc=', score)
        score_list.append(score)



if __name__ == '__main__':
    root = './Dataset/'
    im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print(img_path)

    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)
    n_row, n_column, n_band = img.shape
    X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))

    X_train = np.reshape(X_img, (n_row * n_column, n_band))
    print('training img shape: ', X_train.shape)

    LR, BATCH_SIZE, EPOCH = 0.00002, 64, 100
    N_BAND = 5
    acnn = BS_Net_FC(LR, BATCH_SIZE, EPOCH, N_BAND)
    acnn.fit(X_train, img=X_img, gt=gt)


