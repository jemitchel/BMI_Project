import os
from joblib import load


def loadup(dir):

    os.chdir(dir)

    clf_gene = load("clf_gene4.joblib")
    fea_gene = load("fea_gene4.joblib")
    clf_miRNA = load("clf_miRNA4.joblib")
    fea_miRNA = load("fea_miRNA4.joblib")
    clf_meth = load("clf_meth4.joblib")
    fea_meth = load("fea_meth4.joblib")
    clf_CNV = load("clf_CNV4.joblib")
    fea_CNV = load("fea_CNV4.joblib")
    clf = load("clf4.joblib")

    return (clf_gene, fea_gene, clf_miRNA, fea_miRNA, clf_meth, fea_meth, clf_CNV, fea_CNV, clf)