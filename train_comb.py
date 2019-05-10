from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from feat_select import select_features
from feat_select import discretize
import pandas as pd
import csv
import numpy as np
import random
import os
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from sklearn.naive_bayes import ComplementNB


# # loads precomputed features and response
# os.chdir("C:\\Users\\jonat\\Documents\\Spring 2019 Classes\\4813\\Processed_data")
# new_feats = pd.read_csv('new_feats.csv', index_col=0)
# new_feats_labels = pd.read_csv('new_feats_labels.csv', index_col=0)

# outputs a classifier and optimal features
def tr_comb(X,y):

    # # normalizing gene expression and miRNA datasets
    # X_copy = pd.DataFrame(X, copy=True) # copies the original dataframe
    # X_scaler = preprocessing.MinMaxScaler().fit(X)
    # X = X_scaler.transform(X)
    # X = pd.DataFrame(X,columns=list(X_copy)).set_index(X_copy.index.values)

    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=12)
    # parameters to test
    kernels = ['linear', 'rbf', 'sigmoid']
    c_values = [.001,.01,0.1, 1, 10,50,100,250,500,1000,2500,5000,10000]
    # c_values = [.1,1,10,50]

    para_list = [(k, c) for k in kernels for c in c_values]
    tot_acc = []
    best_params = []
    for (k, c) in para_list:
        acc = []
        for train, test in kf.split(X, y):
            tr_ndx = X.index.values[train]
            te_ndx = X.index.values[test]
            X_train, X_test = X.loc[tr_ndx, :], X.loc[te_ndx, :]
            y_train, y_test = y.loc[tr_ndx, :], y.loc[te_ndx, :]

            # start of classification
            clf = svm.SVC(C=c, gamma="auto", kernel=k)
            clf.fit(X_train, y_train.values.ravel())
            # acc.append(clf.score(X_test,y_test))
            # fsc = f1_score(y_test, clf.predict(X_test))
            c_kap = cohen_kappa_score(y_test, clf.predict(X_test))
            # acc.append(fsc)
            acc.append(c_kap)
        tot_acc.append(np.mean(acc))
        best_params.append([k, c])
        # print(np.mean(acc))
        # print([k,c])
    # print(max(tot_acc))
    ndx = np.argmax(tot_acc)
    final_pset = best_params[ndx]
    # print(final_pset)
    clf = svm.SVC(C=final_pset[1], gamma="auto", kernel=final_pset[0], probability=True)
    clf.fit(X, y.values.ravel())

    return (clf)

def maj_vote(X,y):
    # this run directly with the test set. no cross validation used
    n_modes = X.shape[1]
    pred = []
    for i in range(X.shape[0]):
        tot = 0
        for j in range(X.shape[1]):
            if X.iloc[i,j] < 0:
                tot += 1
        if tot >= n_modes/2:
            pred.append(0)
        else:
            pred.append(1)

    correct = 0
    for i in range(len(pred)):
        if pred[i] == y.iloc[i,0]:
            correct += 1

    print(pred)
    print(y)
    score = correct/len(pred)
    # print(score)

    return score

def weight_vote(X,y,weights):
    # this run directly with the test set. no cross validation used
    n_modes = X.shape[1]
    pred = []
    for i in range(X.shape[0]):
        tot = 0
        for j in range(X.shape[1]):
            tot += weights[j]*X.iloc[i,j]
        if tot < 0:
            pred.append(0)
        else:
            pred.append(1)

    correct = 0
    for i in range(len(pred)):
        if pred[i] == y.iloc[i,0]:
            correct += 1

    print(pred)
    print(y)
    score = correct/len(pred)
    print(score)

def tr_comb_grid(X,y):

    # # normalizing gene expression and miRNA datasets
    # X_copy = pd.DataFrame(X, copy=True) # copies the original dataframe
    # X_scaler = preprocessing.MinMaxScaler().fit(X)
    # X = X_scaler.transform(X)
    # X = pd.DataFrame(X,columns=list(X_copy)).set_index(X_copy.index.values)

    c_kap = make_scorer(cohen_kappa_score)
    parameters = {'kernel': ('linear','poly','rbf','sigmoid'), 'C': [.001,.005,.01,.05,0.1,.25,.5,1,1.25,1.5,2,2.5,3,4,5,10],'gamma':[.01,.1,.25,.5,.75,1,1.25,1.5,2]}
    # parameters = {'kernel': ['linear'], 'C': [.001,.005,.01,.05,0.1,.25,.5,1,1.25,1.5,2,2.5,3,4,5,10],'gamma':[.01,.1,.25,.5,.75,1,1.25,1.5,2]}
    svc = svm.SVC(gamma="auto",class_weight='balanced')
    clf = GridSearchCV(svc, parameters, cv=4,iid=False,scoring=c_kap)
    # clf = RandomizedSearchCV(svc, parameters, cv=4,iid=False,n_iter=75)
    clf.fit(X,y.values.ravel())
    print(clf.best_params_)
    clf = svm.SVC(gamma=clf.best_params_['gamma'],class_weight='balanced',C=clf.best_params_['C'],kernel=clf.best_params_['kernel'],probability=True)
    clf.fit(X, y.values.ravel())
    # print(clf.coef_)
    return clf

def bayes(X,y):
    clf = ComplementNB()
    clf.fit(X, y.values.ravel())
    return clf



