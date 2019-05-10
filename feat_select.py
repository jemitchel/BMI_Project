import pandas as pd
import pymrmr
import numpy as np
from scipy.stats import ttest_ind
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold

def select_features(X,y,modality,method,n_feats):

    if method == 'mrmr':
        if modality == 'gene' or modality == 'meth': #for these doing prefiltering with ttest
            init_feats = reduce(X, y, 2000)
            X = X.loc[:, init_feats]
        elif modality == 'CNV':
            X, y = discretize(X, y, modality, 1)
            init_feats = chi(X, y, 2000)
            X = X.loc[:, init_feats]

        # calls helper function to discretize
        X,y = discretize(X,y,modality,.5) #4th param is number std away from mean as discretization threshold
        z = pd.concat([y, X], axis=1)

        # calling mRMR function
        feat_selected = pymrmr.mRMR(z,'MIQ',n_feats)
    elif method == 'ttest':
        feat_selected = reduce(X, y, n_feats)
    elif method == 'chi-squared':
        X,y = discretize(X,y,modality,.3)
        feat_selected = chi(X, y, n_feats)
    elif method == 'minfo':
        if modality == 'miRNA':
            X, y = discretize(X, y, modality, 2)
        elif modality == 'gene' or modality == 'meth':
            X, y = discretize(X, y, modality, 1)
            init_feats = chi(X, y, 5000)
            X = X.loc[:, init_feats]
        elif modality == 'CNV':
            X, y = discretize(X, y, modality, 1)
            init_feats = chi(X, y, 1000)
            X = X.loc[:, init_feats]

        feat_selected = minfo(X,y,n_feats)

    return (feat_selected)

def discretize(X,y,modality,n): #features need to be -2,0,2 and response needs to be -1,1
    # discretizes feature data
    if modality == 'CNV':
        X2 = pd.DataFrame(X, copy=True)
        X[X2 == -1] = 0
        X[X2 == 0] = 1
        X[X2 == 1] = 2
    else:

        std = X.std(axis=0) # for using non trimmed std
        av = X.mean(axis=0) # for using non trimmed mean

        X[X < av - (n * std)] = 0
        X[X > av + (n * std)] = 2
        X[abs(X) != 2] = 4
        X = X.astype('int64') #makes the numbers integers, not floats

    # changes discretization for class labels
    y[y == 0] = -1
    y = y.astype('int64')  # makes the numbers integers, not floats
    return (X,y)

def reduce(X,y,n_feats):
    t1 = X.loc[y['label'] == 0]
    t2 = X.loc[y['label'] == 1]
    stat,pv = ttest_ind(t1,t2,axis=0)
    feats = []
    indicies = np.argsort(pv)
    for i in range(len(indicies)):
        if i < n_feats:
            feats.append(list(X)[indicies[i]])
    return feats

def chi(X,y,n_feats):
    _,pvals = chi2(X,y)
    feats = []
    indicies = np.argsort(pvals)
    for i in range(len(indicies)):
        if i < n_feats:
            feats.append(list(X)[indicies[i]])
    return feats


def minfo(X,y,n_feats):
    # score = mutual_info_classif(X,y.values.ravel(),random_state=42)
    score = mutual_info_classif(X,y.values.ravel(),discrete_features=True)
    feats = []
    indicies = np.argsort(score)
    indicies = indicies[::-1]
    for i in range(len(indicies)):
        if i < n_feats:
            feats.append(list(X)[indicies[i]])
        else:
            break
    return feats

