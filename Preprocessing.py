# coding: utf-8

from __future__ import print_function
import numpy as np
import spectral as spy


class Processor:
    def __init__(self):
        pass

    def prepare_data(self, img_path, gt_path):
        if img_path[-3:] == 'mat':
            import scipy.io as sio
            img_mat = sio.loadmat(img_path)
            gt_mat = sio.loadmat(gt_path)
            img_keys = img_mat.keys()
            gt_keys = gt_mat.keys()
            img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']
            gt_key = [k for k in gt_keys if k != '__version__' and k != '__header__' and k != '__globals__']
            return img_mat.get(img_key[0]).astype('float64'), gt_mat.get(gt_key[0]).astype('int8')
        else:
            import spectral as spy
            img = spy.open_image(img_path).load()
            gt = spy.open_image(gt_path)
            a = spy.principal_components()
            a.transform()
            return img, gt.read_band(0)

    def get_correct(self, img, gt):
        """
        :param img: 3D arr
        :param gt: 2D arr
        :return: covert arr  [n_samples,n_bands]
        """
        gt_1D = gt.reshape(-1)
        index = gt_1D.nonzero()
        print("nonzero",len(index[0]))
        gt_correct = gt_1D[index]
        img_2D = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
        img_correct = img_2D[index]
        return img_correct, gt_correct

    def get_tr_tx_index(self, y, test_size=0.9):
        from sklearn.model_selection import train_test_split
        X_train_index, X_test_index, y_train_, y_test_ = \
            train_test_split(np.arange(0, y.shape[0]), y, test_size=test_size)
        return X_train_index, X_test_index


    def split_each_class(self, X, y, each_train_size=10):
        X_tr, y_tr, X_ts, y_ts = [], [], [], []
        for c in np.unique(y):
            y_index = np.nonzero(y == c)[0]
            np.random.shuffle(y_index)
            cho, non_cho = np.split(y_index, [each_train_size, ])
            X_tr.append(X[cho])
            y_tr.append(y[cho])
            X_ts.append(X[non_cho])
            y_ts.append(y[non_cho])
        X_tr, X_ts, y_tr, y_ts = np.asarray(X_tr), np.asarray(X_ts), np.asarray(y_tr), np.asarray(y_ts)
        return X_tr.reshape(X_tr.shape[0] * X_tr.shape[1], X.shape[1]),\
               X_ts.reshape(X_ts.shape[0] * X_ts.shape[1], X.shape[1]), \
               y_tr.flatten(), y_ts.flatten()

    def stratified_train_test_index(self, y, train_size):
        """
        :param y: labels
        :param train_size: int, absolute number for each classes; float [0., 1.], percentage of each classes
        :return:
        """
        train_idx, test_idx = [], []
        for i in np.unique(y):
            idx = np.nonzero(y == i)[0]
            np.random.shuffle(idx)
            num = np.sum(y == i)
            if 0. < train_size < 1.:
                train_size_ = int(np.ceil(train_size * num))
            elif train_size > num or train_size <= 0.:
                raise Exception('Invalid training size.')
            else:
                train_size_ = np.copy(train_size)
            train_idx += idx[:train_size_].tolist()
            test_idx += idx[train_size_:].tolist()
        train_idx = np.asarray(train_idx).reshape(-1)
        test_idx = np.asarray(test_idx).reshape(-1)
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        return train_idx, test_idx

    def save_experiment(self, y_pre, y_test, file_neme=None, parameters=None):
        """
        save classification results and experiment parameters into files for k-folds cross validation.
        :param y_pre:
        :param y_test:
        :param parameters:
        :return:
        """
        import os
        home = os.getcwd() + '/experiments'
        if not os.path.exists(home):
            os.makedirs(home)
        if parameters == None:
            parameters = [None]
        if file_neme == None:
            file_neme = home + '/scores.npz'
        else:
            file_neme = home + '/' + file_neme + '.npz'

        '''save results and scores into a numpy file'''
        ca, oa, aa, kappa = [], [], [], []
        if np.array(y_pre).shape.__len__() > 1:  # that means test data tested k times
            for y in y_pre:
                ca_, oa_, aa_, kappa_ = self.score(y_test, y)
                ca.append(ca_), oa.append(oa_), aa.append(aa_), kappa.append(kappa_)
        else:
            ca, oa, aa, kappa = self.score(y_test, y_pre)
        np.savez(file_neme, y_test=y_test, y_pre=y_pre, CA=np.array(ca), OA=np.array(oa), AA=aa, Kappa=kappa,
                 param=parameters)
        print('the experiments have been saved in experiments/scores.npz')


    def score(self, y_test, y_predicted):
        from sklearn.metrics import accuracy_score
        '''overall accuracy'''
        oa = accuracy_score(y_test, y_predicted)
        '''average accuracy for each classes'''
        n_classes = max([np.unique(y_test).__len__(), np.unique(y_predicted).__len__()])
        ca = []
        for c in np.unique(y_test):
            y_c = y_test[np.nonzero(y_test == c)]  # find indices of each classes
            y_c_p = y_predicted[np.nonzero(y_test == c)]
            acurracy = accuracy_score(y_c, y_c_p)
            ca.append(acurracy)
        ca = np.array(ca)
        aa = ca.mean()

        '''kappa'''
        kappa = self.kappa(y_test, y_predicted)
        return ca, oa, aa, kappa

    def result2gt(self, y_predicted, test_indexes, gt):
        """

        :param y_predicted:
        :param test_indexes: indexes got from ground truth
        :param gt: 2-dim img
        :return:
        """
        n_row, n_col = gt.shape
        gt_1D = gt.reshape((n_row * n_col))
        gt_1D[test_indexes] = y_predicted
        return gt_1D.reshape(n_row, n_col)


    def pca_transform(self, n_components, samples):
        """

        :param n_components:
        :param samples: [nb_samples, bands]/or [n_row, n_column, n_bands]
        :return:
        """
        HSI_or_not = samples.shape.__len__() == 3  # denotes HSI data
        n_row, n_column, n_bands = 0, 0, 0
        if HSI_or_not:
            n_row, n_column, n_bands = samples.shape
            samples = samples.reshape((n_row * n_column, n_bands))
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        trans_samples = pca.fit_transform(samples)
        if HSI_or_not:
            return trans_samples.reshape((n_row, n_column, n_components))
        return trans_samples

    def normlize_HSI(self, img):
        from sklearn.preprocessing import normalize
        n_row, n_column, n_bands = img.shape
        norm_img = normalize(img.reshape(n_row * n_column, n_bands))
        return norm_img.reshape(n_row, n_column, n_bands)

    def each_class_OA(self, y_test, y_predicted):
        """
        get each OA for all classes respectively
        :param y_test:
        :param y_predicted:
        :return:{}
        """
        classes = np.unique(y_test)
        results = []
        for c in classes:
            y_c = y_test[np.nonzero(y_test == c)]  # find indices of each classes
            y_c_p = y_predicted[np.nonzero(y_test == c)]
            acurracy = self.score(y_c, y_c_p)
            results.append(acurracy)
        return np.array(results)

    def kappa(self, y_test, y_predicted):
        from sklearn.metrics import cohen_kappa_score
        return round(cohen_kappa_score(y_test, y_predicted), 3)


    def get_tr_ts_index_num(self, y, n_labeled=10):
        import random
        classes = np.unique(y)
        X_train_index, X_test_index = np.empty(0, dtype='int8'), np.empty(0, dtype='int8')
        for c in classes:
            index_c = np.nonzero(y == c)[0]
            random.shuffle(index_c)
            X_train_index = np.append(X_train_index, index_c[:n_labeled])
            X_test_index = np.append(X_test_index, index_c[n_labeled:])
        return X_train_index, X_test_index

    def save_res_4kfolds_cv(self, y_pres, y_tests, file_name=None, verbose=False):
        ca, oa, aa, kappa = [], [], [], []
        for y_p, y_t in zip(y_pres, y_tests):
            ca_, oa_, aa_, kappa_ = self.score(y_t, y_p)
            ca.append(np.asarray(ca_)), oa.append(np.asarray(oa_)), aa.append(np.asarray(aa_)),
            kappa.append(np.asarray(kappa_))
        ca = np.asarray(ca) * 100
        oa = np.asarray(oa) * 100
        aa = np.asarray(aa) * 100
        kappa = np.asarray(kappa)
        ca_mean, ca_std = np.round(ca.mean(axis=0), 2), np.round(ca.std(axis=0), 2)
        oa_mean, oa_std = np.round(oa.mean(), 2), np.round(oa.std(), 2)
        aa_mean, aa_std = np.round(aa.mean(), 2), np.round(aa.std(), 2)
        kappa_mean, kappa_std = np.round(kappa.mean(), 3), np.round(kappa.std(), 3)

        cas = np.asarray([ca_mean, ca_std])
        aas = np.asarray([aa_mean, aa_std])
        oas =  np.asarray([oa_mean, oa_std])
        kappas = np.asarray([kappa_mean, kappa_std])


        return cas, aas, oas, kappas

    # def view_clz_map(self, gt, y_index, y_predicted, save_path=None, show_error=False):
    #     """
    #     view HSI classification results
    #     :param gt:
    #     :param y_index: index of excluding 0th classes
    #     :param y_predicted:
    #     :param show_error:
    #     :return:
    #     """
    #     n_row, n_column = gt.shape
    #     gt_1d = gt.reshape(-1).copy()
    #     nonzero_index = gt_1d.nonzero()
    #     gt_corrected = gt_1d[nonzero_index]
    #     if show_error:
    #         t = y_predicted.copy()
    #         correct_index = np.nonzero(y_predicted == gt_corrected[y_index])
    #         t[correct_index] = 0  # leave error
    #         gt_corrected[:] = 0
    #         gt_corrected[y_index] = t
    #         gt_1d[nonzero_index] = t
    #     else:
    #         gt_corrected[y_index] = y_predicted
    #         gt_1d[nonzero_index] = gt_corrected
    #     gt_map = gt_1d.reshape((n_row, n_column)).astype('uint8')
    #     spy.imshow(classes=gt_map)
    #     if save_path != None:
    #         spy.save_rgb(save_path, gt_map, colors=spy.spy_colors)
    #         print('the figure is saved in ', save_path)

    def split_source_target(self, X, y, split_attribute_index, split_threshold, save_name=None):
        """
        split source/target domain data for transfer learning according to attribute
        :param X:
        :param y:
        :param split_attribute_index:
        :param split_threshold: split condition. e.g if 1.2 those x[:,index] >= 1.2 are split into source
        :param save_name:
        :return:
        """
        source_index = np.nonzero(X[:, split_attribute_index] >= split_threshold)
        target_index = np.nonzero(X[:, split_attribute_index] < split_threshold)
        X_source = X[source_index]
        X_target = X[target_index]
        y_source = y[source_index].astype('int')
        y_target = y[target_index].astype('int')
        if save_name is not None:
            np.savez(save_name, X_source=X_source, X_target=X_target, y_source=y_source, y_target=y_target)
        return X_source, X_target, y_source, y_target

    def results_to_cvs(self, res_file_name, save_name):
        import csv
        dt = np.load(res_file_name)
        ca_mean = np.round(dt['CA'].mean(axis=0) * 100, 2)
        ca_std = np.round(dt['CA'].std(axis=0), 2)
        oa_mean = np.round(dt['OA'].mean() * 100, 2)
        oa_std = np.round(dt['OA'].std(axis=0), 2)
        aa_mean = np.round(dt['AA'].mean() * 100, 2)
        aa_std = np.round(dt['AA'].std(axis=0), 2)
        kappa_mean = np.round(dt['Kappa'].mean(), 3)
        kappa_std = np.round(dt['Kappa'].std(axis=0), 2)
        with open(save_name, 'wb') as f:
            writer = csv.writer(f)
            for i in zip(ca_mean, ca_std):
                writer.writerow(i)
            writer.writerow([oa_mean, oa_std])
            writer.writerow([aa_mean, aa_std])
            writer.writerow([kappa_mean, kappa_std])

    def view_clz_map_spyversion4single_img(self, gt, y_test_index, y_predicted, save_path=None, show_error=False,
                                           show_axis=False):
        """
        view HSI classification results
        :param gt:
        :param y_test_index: test index of excluding 0th classes
        :param y_predicted:
        :param show_error:
        :return:
        """
        n_row, n_column = gt.shape
        gt_1d = gt.reshape(-1).copy()
        nonzero_index = gt_1d.nonzero()
        gt_corrected = gt_1d[nonzero_index]
        if show_error:
            t = y_predicted.copy()
            correct_index = np.nonzero(y_predicted == gt_corrected[y_test_index])
            t[correct_index] = 0  # leave error
            gt_corrected[:] = 0
            gt_corrected[y_test_index] = t
            gt_1d[nonzero_index] = t
        else:
            gt_corrected[y_test_index] = y_predicted
            gt_1d[nonzero_index] = gt_corrected
        gt_map = gt_1d.reshape((n_row, n_column)).astype('uint8')
        spy.imshow(classes=gt_map)
        if save_path != None:
            import matplotlib.pyplot as plt
            spy.save_rgb('temp.png', gt_map, colors=spy.spy_colors)
            if show_axis:
                plt.savefig(save_path, format='eps', bbox_inches='tight')
            else:
                plt.axis('off')
                plt.savefig(save_path, format='eps', bbox_inches='tight')
            # self.classification_map(gt_map, gt, 24, save_path)
            print('the figure is saved in ', save_path)


    def standardize_label(self, y):
        """
        standardize the classes label into 0-k
        :param y: 
        :return: 
        """
        import copy
        classes = np.unique(y)
        standardize_y = copy.deepcopy(y)
        for i in range(classes.shape[0]):
            standardize_y[np.nonzero(y == classes[i])] = i
        return standardize_y

    def one2array(self, y):
        n_classes = np.unique(y).__len__()
        y_expected = np.zeros((y.shape[0], n_classes))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

    def zca_whitening(self, x, epsilon=1e-6, mean=None, whitening=None):
        '''
        Applies ZCA whitening the the input data.
        Arguments:
            x: numpy array of shape (batch_size, dim). If the input has
                more than 2 dimensions (such as images), it will be flatten the
                data.
            epsilon: an hyper-parameter called the whitening coefficient, default is 1e-6
            mean: numpy array of shape (dim) that will be used as the mean.
                If None (Default), the mean will be computed from the input data.
            whitening: numpy array shaped (dim, dim) that will be used as the
                whitening matrix. If None (Default), the whitening matrix will be
                computed from the input data.
        Returns:
            white_data: numpy array with whitened data. Has the same shape as
                the input.
            mean: numpy array of shape (dim) that contains the mean of each input
                dimension. If mean was provided as input, this is a copy of it.
            whitening:  numpy array of shape (dim, dim) that contains the whitening
                matrix. If whitening was provided as input, this is a copy of it.
        '''
        if not x.size:
            # Simply return if data_set is empty
            return x, mean, whitening
        data_shape = x.shape
        size = data_shape[0]
        white_data = x.reshape((size, -1))

        if mean is None:
            # No mean matrix, we must compute it
            mean = white_data.mean(axis=0)
        # Remove mean
        white_data -= mean

        # If no whitening matrix, we must compute it
        if whitening is None:
            cov = np.dot(white_data.T, white_data) / size
            U, S, V = np.linalg.svd(cov)
            whitening = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + epsilon))), U.T)

        white_data = np.dot(white_data, whitening)
        return white_data.reshape(data_shape), mean, whitening