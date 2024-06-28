from Preprocessing import Processor
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split

def eval_band_cv(X, y, times=10, test_size=0.95):
    print(X.shape)
    p = Processor()
    estimator = SVC(C=1e5, kernel='rbf', gamma=1.)
    estimator_pre, y_test_all = [], []
    for i in range(times):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=None, shuffle=True, stratify=y)
        y_test_all.append(y_test)
        estimator.fit(X_train, y_train)
        y_pre = estimator.predict(X_test)
        estimator_pre.append(y_pre)
    score_dic = {'ca': [], 'oa': [], 'aa': [], 'kappa': [] }
    ca, oa, aa, kappa = p.save_res_4kfolds_cv(estimator_pre, y_test_all, file_name=None, verbose=False)
    score_dic['ca'] = ca
    score_dic['oa'] = oa
    score_dic['aa'] = aa
    score_dic['kappa'] = kappa
    return score_dic
