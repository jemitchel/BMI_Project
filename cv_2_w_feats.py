from cv_2 import tr
import pandas as pd
from feat_select import select_features
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# this does cross validation and iterates over different number of features
def do_cv(X, y, X_test, y_test, f_sel, modality,n_feats,split,make_fplots):
    max_n_feats = n_feats
    feat_iters = []
    X2 = pd.DataFrame(X, copy=True)  # copies the original feature dataframe
    y2 = pd.DataFrame(y, copy=True)  # copies the original feature dataframe
    feat_selected = select_features(X, y, modality, f_sel, max_n_feats)


    test_acc = []
    train_acc = []
    test_auc = []
    train_auc = []
    num_feats = []

    res = []
    best_clfs = []
    best_feats = []
    best_params = []
    hmap_data = np.array(['kernel','C','num_feats','score'])

    for i in range(2, max_n_feats, split):
        print(i)
        num_feats.append(i)
        X3 = pd.DataFrame(X2, copy=True)  # copies the original feature dataframe
        y3 = pd.DataFrame(y2, copy=True)  # copies the original feature dataframe
        clf, fea_ = tr(X3, y3, modality, f_sel, 'none', feat_selected[0:i])

        res.append(clf.best_score_)
        best_feats.append(fea_)
        best_params.append([clf.best_params_,i])
        for param, score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
            hmap_data = np.vstack((hmap_data,np.array([param['kernel'],param['C'],i,score])))

        # do this is picking clf from f1 score or cohens kappa. makes it give an accuracy output
        X5 = X2[fea_]
        clf = svm.SVC(gamma='auto',class_weight='balanced',C=clf.best_params_['C'],kernel=clf.best_params_['kernel'],probability=True)
        clf.fit(X5, y2.values.ravel())
        best_clfs.append(clf) #this is list from which final clf is selected


        # stuff for test
        X4 = X2[fea_]
        X_test2 = X_test[fea_]
        train_acc.append(clf.score(X4, y2))
        test_acc.append(clf.score(X_test2, y_test))
        c1, c2, _ = roc_curve(y2.values.ravel(), clf.decision_function(X4).ravel())
        train_auc.append(auc(c1, c2))
        c1, c2, _ = roc_curve(y_test.values.ravel(), clf.decision_function(X_test2).ravel())
        test_auc.append(auc(c1, c2))

    print(res)
    print(best_feats)
    ndx = np.argmax(res)
    print(hmap_data)
    print('acc',test_acc[ndx])
    print('auc',test_auc[ndx])
    print(best_params[ndx])
    print('max',res[ndx])

    if make_fplots == True:
        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.6, 0.85, .2], ylim=(0, 1))
        ax2 = fig.add_axes([0.1, 0.1, 0.85, .2], ylim=(0, 1))

        ax1.plot(num_feats, train_acc, 'r',label='train')
        ax1.plot(num_feats, test_acc, 'b',label='test')
        ax1.set_title(modality + ' Accuracy', fontsize=15)
        ax1.set_xlabel('Number of Features', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.legend(loc='bottom left')

        ax2.plot(num_feats, train_auc, 'r')
        ax2.plot(num_feats, test_auc, 'b')
        ax2.set_title(modality + ' AUC', fontsize=15)
        ax2.set_xlabel('Number of Features', fontsize=10)
        ax2.set_ylabel('AUC', fontsize=10)

        plt.show()

    return best_clfs[ndx],best_feats[ndx],hmap_data



