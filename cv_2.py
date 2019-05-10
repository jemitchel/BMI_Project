import pandas as pd
from feat_select import select_features
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
import random
from sklearn.metrics import roc_curve, auc


def tr(X,y,type,f_sel,n_feats,feat_selected):
    fsc = make_scorer(f1_score)
    c_kap = make_scorer(cohen_kappa_score)

    # need to put everything below in a loop so can test different feature set sizes
    X2 = pd.DataFrame(X, copy=True) # copies the original feature dataframe
    y2 = pd.DataFrame(y, copy=True) # copies the original feature dataframe

    # selects top n features
    if feat_selected == 'none':
        feat_selected = select_features(X, y, type, f_sel, n_feats)
        X2 = X2[feat_selected] # shrinks feature matrix to only include selected features
    else:
        X2 = X2[feat_selected]

    # start of classification
    parameters = {'kernel': ('linear','poly','rbf','sigmoid'), 'C': [.001,.005,.1,.5,1,1.5,2,2.5,3,4,5,6,7,8,9,10,11,12,15,20,25,50,75,100,150,200,250]}
    svc = svm.SVC(gamma="auto",probability=True,class_weight='balanced')
    if type == 'miRNA':
        clf = GridSearchCV(svc, parameters, cv=4,scoring=c_kap,iid=False)
    else:
        clf = GridSearchCV(svc, parameters, cv=4,iid=False)

    clf.fit(X2, y2.values.ravel()) #can also try X here if alter it first

    return clf, feat_selected

