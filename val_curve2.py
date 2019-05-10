from train_ind import tr_ind
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from cv_2_w_feats import do_cv


def gen_curve(frame,lbl,modality,n_times):
    pt_x = []
    pt_y = []
    all_auc = []
    seeds = [2301,3990,8490,8408,9084,2736,111,9483,345,1220]
    for i in range(n_times):
        seed = seeds[i]
        train_labels, test_labels, train_class, test_class = train_test_split(
            lbl['case_id'], lbl, test_size=0.15, random_state=seed)

        X_train = frame[train_labels].T
        X_test = frame[test_labels].T

        y_train = train_class.set_index('case_id')  # changes first column to be indices
        y_test = test_class.set_index('case_id')  # changes first column to be indices

        if modality == 'gene' or modality == 'miRNA':
            # normalizing gene expression and miRNA datasets
            X_train_copy = pd.DataFrame(X_train, copy=True)  # copies the original dataframe
            scaler = preprocessing.MinMaxScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_train = pd.DataFrame(X_train, columns=list(X_train_copy)).set_index(X_train_copy.index.values)

            X_test_copy = pd.DataFrame(X_test, copy=True)  # copies the original dataframe
            X_test = scaler.transform(X_test)
            X_test = pd.DataFrame(X_test, columns=list(X_test_copy)).set_index(X_test_copy.index.values)

        X_train_copy = pd.DataFrame(X_train, copy=True)
        y_train_copy = pd.DataFrame(y_train, copy=True)

        if modality == 'gene':
            clf, feats, _ = do_cv(X_train_copy, y_train_copy, X_test, y_test, 'ttest', 'gene', 40, 2)

        elif modality == 'CNV':
            clf, feats, _ = do_cv(X_train_copy, y_train_copy, X_test, y_test, 'minfo', 'CNV', 50, 2)

        elif modality == 'meth':
            clf, feats, _ = do_cv(X_train_copy, y_train_copy, X_test, y_test, 'minfo', 'meth', 60, 2)

        else:
            clf, feats, _ = do_cv(X_train_copy, y_train_copy, X_test, y_test, 'minfo', 'miRNA',24, 2)


        X_test = X_test[feats]
        X_train = X_train[feats]
        test_acc = clf.score(X_test,y_test)
        train_acc = clf.score(X_train,y_train)

        pt_x.append(train_acc)
        pt_y.append(test_acc)
        print(clf.decision_function(X_test))
        c1, c2, _ = roc_curve(y_test.values.ravel(), clf.decision_function(X_test).ravel())
        all_auc.append(auc(c1,c2))

    print(all_auc)
    plt.scatter(pt_x,pt_y)
    x = np.linspace(0, 1, 100)
    plt.plot(x,x)
    plt.xlabel('Training Accuracy',fontsize=15)
    plt.ylabel('Test Set Accuracy',fontsize=15)
    # plt.tick_params(labelsize = 10)
    name = ''
    if modality == 'gene':
        name = 'Gene Expression'
    elif modality == 'CNV':
        name = 'CNV'
    elif modality == 'meth':
        name = 'Methylation'
    elif modality == 'miRNA':
        name = 'miRNA'
    plt.title(name + ' Validation Plot',fontsize=20)
    plt.show()


