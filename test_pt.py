import pandas as pd
import os
from joblib import load
from sklearn import svm
from sklearn import preprocessing
import numpy as np


def test_pt(dir, pt):

    # os.chdir("C:\\Users\\jonat\\Documents\\Spring 2019 Classes\\4813\\BHI_data"")
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
    print(clf_gene)

    geneNames = pd.read_csv("pr_coding_feats.csv")

    # test_pt = 4a831893-f4ca-4357-a756-b5e954e35dd7 -- actual survival: 0
    ptGene = pd.read_csv('%s_gene.csv' % pt)
    ptGene["gene_id"] = [word[0:15] for word in ptGene["gene_id"]]
    ptGene = ptGene.set_index("gene_id")
    ptmiRNA = pd.read_csv('%s_miRNA.csv' % pt)
    ptmiRNA = ptmiRNA.set_index("miRNA_ID")
    ptmeth = pd.read_csv('%s_meth.csv' % pt)
    ptmeth = ptmeth.set_index("Composite Element REF")
    # ptCNV = pd.read.csv("%s_CNV.csv" % pt)
    # ptCNV

    ptGene = ptGene.loc[fea_gene,:]
    ptGene_scaler = preprocessing.MinMaxScaler().fit(ptGene)
    ptGene_scaled = ptGene_scaler.transform(ptGene)

    ptmiRNA = ptmiRNA.loc[fea_miRNA, :]
    ptmiRNA_scaler = preprocessing.MinMaxScaler().fit(ptmiRNA)
    ptmiRNA_scaled = ptmiRNA_scaler.transform(ptmiRNA)

    ptmeth = ptmeth.loc[fea_meth, :]

    print(ptGene_scaled)
    print(clf_gene.predict_proba(ptGene_scaled.T))
    print(ptmiRNA_scaled)
    print(clf_miRNA.predict_proba(ptmiRNA_scaled.T))
    print(ptmeth)
    print(clf_meth.predict_proba(ptmeth.T))
    pt_gene_pred = clf_gene.predict_proba(ptGene_scaled.T)
    pt_miRNA_pred = clf_miRNA.predict_proba(ptmiRNA_scaled.T)
    pt_meth_pred = clf_meth.predict_proba(ptmeth.T)
    pt_df_in = {'meth':[pt_meth_pred[0][1]], 'miRNA':[pt_miRNA_pred[0][1]], 'gene':[pt_gene_pred[0][1]]}
    pt_df = pd.DataFrame(pt_df_in)
    print(pt_df_in)
    print(pt_df)
    # print(pt_df)
    # print(clf.predict(pt_df))
    # print(clf.predict_proba(pt_df)[0][0])
    # print([pt_miRNA_pred[0][0], pt_gene_pred[0][0]])
    # print(clf.predict([[pt_miRNA_pred[0][0]], [pt_gene_pred[0][0]]]))


    ## CHANGE TO ACTUALLY WORK
    pred_out = ''
    prob_out = ''
    pred = clf.predict(pt_df)
    if pred == 1:
        pred_out = 'High Survival (>= 5 years)'
    else:
        pred_out = 'Low Survival (< 5 years)'
    prob = clf.predict_proba(pt_df)[0][0]
    prob_out = '%.2f%%' % (prob * 100)
    print(pred)
    print(prob)
    return(pred_out, prob_out)
    # return(clf_gene, fea_gene, clf_miRNA, fea_miRNA, clf_meth, fea_meth, clf_CNV, fea_CNV, clf)

test_pt("C:\\Users\\jonat\\Documents\\Spring 2019 Classes\\4813\\BHI_data", 'test_pt')