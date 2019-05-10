import operator
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from train_comb import tr_comb_grid
from train_comb import maj_vote
# from train_comb import bayes
from sklearn.metrics import roc_curve, auc
import numpy as np
from val_curve2 import gen_curve
from cv_2_w_feats import do_cv
from joblib import dump, load
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import warnings

# suppresses the warning that pops up when F-score encounters all 1 class prediction
warnings.filterwarnings("ignore")

def pipeline(rem_zeros,compute,make_plot,seed,dir,method='ttest',test_size=0.15):

    # loads all data
    os.chdir(dir)
    labels = pd.read_csv('Label_selected.csv')
    ndx = labels.index[labels['label'] > -1].tolist() #gets indices of patients to use
    lbl = labels.iloc[ndx,[0,4]] #makes a new dataframe of patients to use (their IDs and survival response)
    surv = labels.iloc[ndx,:]
    genes = pd.read_csv('pr_coding_feats.csv')
    genes = genes.set_index('ensembl_gene_id')


    os.chdir(dir)
    gene = pd.read_csv('gene.csv',index_col=0)
    miRNA = pd.read_csv('miRNA.csv',index_col=0)
    meth = pd.read_csv('meth.csv',index_col=0)
    CNV = pd.read_csv('CNV.csv',index_col=0)

    os.chdir(dir)

    # optionally removes rows (features) that are all 0 across patients
    if rem_zeros == True:
        gene = gene.loc[~(gene==0).all(axis=1)]
        miRNA = miRNA.loc[~(miRNA==0).all(axis=1)]

    # splitting labels into train set and validation set
    train_labels, test_labels, train_class, test_class = train_test_split(
        lbl['case_id'], lbl, test_size=test_size, random_state=seed)

    # removes features (rows) that have any na in them
    meth = meth.dropna(axis='rows')
    miRNA = miRNA.dropna(axis='rows')
    gene = gene.dropna(axis='rows')
    CNV = CNV.dropna(axis='rows')


    # divides individual modalities into train and test sets based on same patient splits and transposes
    miRNA_train = miRNA[train_labels].T
    miRNA_test = miRNA[test_labels].T
    gene_train = gene[train_labels].T
    gene_test = gene[test_labels].T
    CNV_train = CNV[train_labels].T
    CNV_test = CNV[test_labels].T
    meth_train = meth[train_labels].T
    meth_test = meth[test_labels].T


    # normalizing gene expression and miRNA datasets
    miRNA_train_copy = pd.DataFrame(miRNA_train, copy=True) # copies the original dataframe
    miRNA_scaler = preprocessing.MinMaxScaler().fit(miRNA_train)
    miRNA_train = miRNA_scaler.transform(miRNA_train)
    miRNA_train = pd.DataFrame(miRNA_train,columns=list(miRNA_train_copy)).set_index(miRNA_train_copy.index.values)

    miRNA_test_copy = pd.DataFrame(miRNA_test, copy=True)  # copies the original dataframe
    miRNA_test = miRNA_scaler.transform(miRNA_test)
    miRNA_test = pd.DataFrame(miRNA_test, columns=list(miRNA_test_copy)).set_index(miRNA_test_copy.index.values)

    gene_train_copy = pd.DataFrame(gene_train, copy=True) # copies the original dataframe
    gene_scaler = preprocessing.MinMaxScaler().fit(gene_train)
    gene_train = gene_scaler.transform(gene_train)
    gene_train = pd.DataFrame(gene_train,columns=list(gene_train_copy)).set_index(gene_train_copy.index.values)

    gene_test_copy = pd.DataFrame(gene_test, copy=True)  # copies the original dataframe
    gene_test = gene_scaler.transform(gene_test)
    gene_test = pd.DataFrame(gene_test, columns=list(gene_test_copy)).set_index(gene_test_copy.index.values)


    train_class = train_class.set_index('case_id') # changes first column to be indices
    test_class = test_class.set_index('case_id') # changes first column to be indices


    # makes copies of the y dataframe because tr_ind alters it
    train_class_copy1,train_class_copy2,train_class_copy3,train_class_copy4,train_class_copy5 = pd.DataFrame(train_class, copy=True),\
                                                                                 pd.DataFrame(train_class, copy=True),\
                                                                                 pd.DataFrame(train_class, copy=True),\
                                                                                 pd.DataFrame(train_class, copy=True),\
                                                                                 pd.DataFrame(train_class, copy=True)



    if make_plot == 'valplot':
        gen_curve(gene,lbl,'gene',10)
        gen_curve(miRNA,lbl,'miRNA',10)
        gen_curve(meth,lbl,'meth',10)
        gen_curve(CNV,lbl,'CNV',10)
        return

    # makes copies of training data
    miRNA_train_copy2 = pd.DataFrame(miRNA_train, copy=True)
    meth_train_copy2 = pd.DataFrame(meth_train, copy=True)
    CNV_train_copy2 = pd.DataFrame(CNV_train, copy=True)
    gene_train_copy2 = pd.DataFrame(gene_train, copy=True)

    if make_plot == 'fplot':
        make_fplot = True
    else:
        make_fplot = False

    if compute == 'recompute':
        # Runs CV script to generate clfs w best parameters
        clf_gene, fea_gene,_ = do_cv(gene_train,train_class_copy1,gene_test,test_class,'ttest','gene',40,2,make_fplot)
        clf_miRNA, fea_miRNA,_ = do_cv(miRNA_train,train_class_copy2,miRNA_test,test_class,'minfo','miRNA',24,2,make_fplot)
        clf_meth, fea_meth,_ = do_cv(meth_train,train_class_copy3,meth_test,test_class,'minfo','meth',60,2,make_fplot)
        clf_CNV, fea_CNV,_ = do_cv(CNV_train,train_class_copy4,CNV_test,test_class,'minfo','CNV',50,2,make_fplot)
    elif compute == 'custom':
        clf_gene, fea_gene, _ = do_cv(gene_train, train_class_copy1, gene_test, test_class, method, 'gene', 100, 2,
                                      make_fplot)
        clf_miRNA, fea_miRNA, _ = do_cv(miRNA_train, train_class_copy2, miRNA_test, test_class, method, 'miRNA', 100, 2,
                                        make_fplot)
        clf_meth, fea_meth, _ = do_cv(meth_train, train_class_copy3, meth_test, test_class, method, 'meth', 100, 2,
                                      make_fplot)
        clf_CNV, fea_CNV, _ = do_cv(CNV_train, train_class_copy4, CNV_test, test_class, method, 'CNV', 100, 2,
                                    make_fplot)
    elif compute == 'precomputed':
        # loads up the precomputed best classifiers and selected features
        clf_gene = load('clf_gene4.joblib')
        fea_gene = load('fea_gene4.joblib')
        clf_meth = load('clf_meth4.joblib')
        fea_meth = load('fea_meth4.joblib')
        clf_CNV = load('clf_CNV4.joblib')
        fea_CNV = load('fea_CNV4.joblib')
        clf_miRNA = load('clf_miRNA4.joblib')
        fea_miRNA = load('fea_miRNA4.joblib')
    else:
        return "enter a valid parameter for compute"

    # shrinks test feature matrix to contain only selected features
    miRNA_test = miRNA_test[fea_miRNA]
    gene_test = gene_test[fea_gene]
    meth_test = meth_test[fea_meth]
    CNV_test = CNV_test[fea_CNV]

    # gets acc results from predicting on test set with individual modalities
    gene_ind_res = clf_gene.score(gene_test,test_class)
    meth_ind_res = clf_meth.score(meth_test,test_class)
    CNV_ind_res = clf_CNV.score(CNV_test,test_class)
    miRNA_ind_res = clf_miRNA.score(miRNA_test,test_class)

    # calculates auc for each modality
    c1_gene, c2_gene, _ = roc_curve(test_class.values.ravel(), clf_gene.decision_function(gene_test).ravel())
    c1_miRNA, c2_miRNA, _ = roc_curve(test_class.values.ravel(), clf_miRNA.decision_function(miRNA_test).ravel())
    c1_CNV, c2_CNV, _ = roc_curve(test_class.values.ravel(), clf_CNV.decision_function(CNV_test).ravel())
    c1_meth, c2_meth, _ = roc_curve(test_class.values.ravel(), clf_meth.decision_function(meth_test).ravel())
    area_gene = auc(c1_gene, c2_gene)
    area_miRNA = auc(c1_miRNA, c2_miRNA)
    area_CNV = auc(c1_CNV, c2_CNV)
    area_meth = auc(c1_meth, c2_meth)

    if make_plot == 'ROC_gene' or make_plot == 'ROC_miRNA' or make_plot == 'ROC_meth' or make_plot == 'ROC_CNV':
        if make_plot == 'ROC_gene':
            c1,c2,area = c1_gene,c2_gene,area_gene
        elif make_plot == 'ROC_miRNA':
            c1,c2,area = c1_miRNA,c2_miRNA,area_miRNA
        elif make_plot == 'ROC_meth':
            c1,c2,area = c1_meth,c2_meth,area_meth
        elif make_plot == 'ROC_CNV':
            c1,c2,area = c1_CNV,c2_CNV,area_CNV

        plt.title('Receiver Operating Characteristic')
        plt.plot(c1, c2, 'b', label='AUC = %0.2f' % area) # change params to change modality
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()




    # START OF DATA INTEGRATION

    fins = []
    areas = []
    fin1s = []
    combinations = [["meth"],["miRNA"],["gene"],["CNV"],["meth","miRNA"],["meth","gene"],["meth","CNV"],
                    ["miRNA","gene"],["miRNA","CNV"],["gene","CNV"],["meth","miRNA","gene"],
                    ["meth","miRNA","CNV"],["miRNA","gene","CNV"],
                    ["meth","gene","CNV"],["meth","miRNA","gene","CNV"]]

    # shrink training feature matricies to selected features
    miRNA_train_copy2 = miRNA_train_copy2[fea_miRNA]
    gene_train_copy2 = gene_train_copy2[fea_gene]
    meth_train_copy2 = meth_train_copy2[fea_meth]
    CNV_train_copy2 = CNV_train_copy2[fea_CNV]

    # gets prediction probabilities for samples in the training data
    pred_miRNA = clf_miRNA.predict_proba(miRNA_train_copy2)[:, 0]
    pred_gene = clf_gene.predict_proba(gene_train_copy2)[:, 0]
    pred_CNV = clf_CNV.predict_proba(CNV_train_copy2)[:, 0]
    pred_meth = clf_meth.predict_proba(meth_train_copy2)[:, 0]

    # creates dataframe with the prediction probabilities
    new_feats = {'sample': miRNA_train.index.values, 'miRNA': pred_miRNA, 'gene': pred_gene, 'meth': pred_meth,
                 'CNV': pred_CNV}
    new_feats = pd.DataFrame(data=new_feats)
    new_feats = new_feats.set_index('sample')

    clfs = []
    cvals = []
    for com in combinations:
        print(com)
        clf = tr_comb_grid(new_feats[com],train_class_copy5)
        # clf = bayes(new_feats[com],train_class_copy5)
        clfs.append(clf)

        pred_miRNA = clf_miRNA.predict_proba(miRNA_test)[:, 0]
        pred_gene = clf_gene.predict_proba(gene_test)[:, 0]
        pred_CNV = clf_CNV.predict_proba(CNV_test)[:, 0]
        pred_meth = clf_meth.predict_proba(meth_test)[:, 0]

        # creates new feature matrix for test predictions
        new_feats_val = {'sample': miRNA_test.index.values, 'miRNA': pred_miRNA, 'gene': pred_gene, 'meth': pred_meth,
                         'CNV': pred_CNV}
        new_feats_val = pd.DataFrame(data=new_feats_val)
        new_feats_val = new_feats_val.set_index('sample')

        fin = clf.score(new_feats_val[com], test_class)
        pred = clf.decision_function(new_feats_val[com])
        c1, c2, _ = roc_curve(test_class.values.ravel(), pred.ravel())
        area = auc(c1, c2)
        cvals.append([c1,c2,area])
        print('auc: ',area)
        print('acc: ',fin)
        areas.append(area)
        fins.append(fin)

    # substitutes list entries for single modalities that were changed during integration
    fins[0] = meth_ind_res
    fins[1] = miRNA_ind_res
    fins[2] = gene_ind_res
    fins[3] = CNV_ind_res
    areas[0] = area_meth
    areas[1] = area_miRNA
    areas[2] = area_gene
    areas[3] = area_CNV
    clfs[0] = clf_meth
    clfs[1] = clf_miRNA
    clfs[2] = clf_gene
    clfs[3] = clf_CNV

    indx = np.argmax(fins)
    tr_score = clfs[indx].score(new_feats[combinations[indx]],train_class_copy5)
    te_score = clfs[indx].score(new_feats_val[combinations[indx]],test_class)

    clf = clfs[indx]

    if make_plot == 'barplot':
        n_groups = 15

        fig, ax = plt.subplots()

        index = np.arange(n_groups)
        bar_width = 0.35

        opacity = 0.4

        rects2 = ax.bar(index, areas, bar_width,
                        alpha=opacity, color='r',
                        label='auc')

        rects3 = ax.bar(index + bar_width, fins, bar_width,
                        alpha=opacity, color='b',
                        label='score')

        ax.set_xlabel('Combination')
        ax.set_ylabel('Scores')

        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(['meth','miRNA','gene','CNV','meth\nmiRNA','meth\ngene','meth\nCNV','miRNA\ngene','miRNA\nCNV',
                            'gene\nCNV','meth\nmiRNA\ngene','meth\nmiRNA\nCNV','miRNA\ngene\nCNV','meth\ngene\nCNV',
                            'meth\nmiRNA\ngene\nCNV'])
        ax.legend()

        fig.tight_layout()
        plt.show()
    elif make_plot == 'ROC_int':
        plt.title('Receiver Operating Characteristic')
        plt.plot(cvals[indx][0], cvals[indx][1], 'b', label='AUC = %0.2f' % cvals[indx][2])
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    elif make_plot == 'KM' or make_plot == 'KM Gene' or make_plot == 'KM miRNA'\
            or make_plot == 'KM Meth' or make_plot == 'KM CNV' or make_plot == 'KM Integrated':
        ax = plt.subplot(111)
        kmf = KaplanMeierFitter()
        m = max(surv["days_to_death"])
        fill_max = {"days_to_death": m}
        surv = surv.fillna(value=fill_max)
        T = surv["days_to_death"]
        surv = surv.replace("alive", False)
        surv = surv.replace("dead", True)
        E = surv["vital_status"]
        if make_plot == 'KM':
            kmf.fit(T, event_observed=E)
            kmf.plot()
            class0 = (surv["label"] == 0)
            kmf.fit(T[class0], event_observed=E[class0], label="Low Survival (<5 years)")
            kmf.plot_survival_function(ax=ax, ci_show=False, fontsize=20)
            kmf.fit(T[~class0], event_observed=E[~class0], label="High Survival (>= 5 years)")
            kmf.plot_survival_function(ax=ax, ci_show=False, fontsize=20)
            ax.set_xlabel("Duration (days)", fontsize=20)
            ax.set_ylabel("Percent Alive", fontsize=20)
            ax.set_title("Breast Cancer Kaplan Meier Survival Curve", fontsize=32)
        elif make_plot == "Kaplan Meier Integrated":
            surv2 = surv.copy(True)
            surv2 = surv2.loc[surv2["case_id"].isin(test_labels.values)]
            surv2 = surv2.set_index("case_id")
            surv2 = surv2.reindex(test_class.index.values)
            prd = clf.predict(new_feats_val[["meth", "miRNA", "gene"]])
            surv2["label_new"] = prd

            count = 0
            preds = clf.predict(new_feats_val[["meth", "miRNA", "gene"]])
            for i in range(test_class.shape[0]):
                if preds[i] == test_class.iloc[i,0]:
                    count += 1
            print(count)

            T2 = surv2["days_to_death"]
            E2 = surv2["vital_status"]
            class02 = (surv2["label_new"] == 0)
            kmf.fit(T2[class02], event_observed=E2[class02], label="Low Survival")
            kmf.plot_survival_function(ax=ax, ci_show=False)
            kmf.fit(T2[~class02], event_observed=E2[~class02], label="High Survival")
            kmf.plot_survival_function(ax=ax, ci_show=False)
            ax.set_xlabel("Duration (days)")
            ax.set_ylabel("Percent Alive")
            ax.set_title("Integrated Classifier Kaplan Meier Survival Curve")
            ax.set_ylim((0,1))

            plt.show()
        elif make_plot == 'KM Gene':
            surv2 = surv.copy(True)
            surv2 = surv2.loc[surv2["case_id"].isin(test_labels.values)]
            surv2 = surv2.set_index("case_id")
            surv2 = surv2.reindex(test_class.index.values)
            prd = clf_gene.predict(gene_test.loc[:,fea_gene])
            surv2["label_new"] = prd

            count = 0
            preds = clf_gene.predict(gene_test.loc[:,fea_gene])
            for i in range(test_class.shape[0]):
                if preds[i] == test_class.iloc[i, 0]:
                    count += 1
            print(count)

            T2 = surv2["days_to_death"]
            E2 = surv2["vital_status"]
            class02 = (surv2["label_new"] == 0)
            kmf.fit(T2[class02], event_observed=E2[class02], label="Low Survival")
            kmf.plot_survival_function(ax=ax, ci_show=False)
            kmf.fit(T2[~class02], event_observed=E2[~class02], label="High Survival")
            kmf.plot_survival_function(ax=ax, ci_show=False)
            ax.set_xlabel("Duration (days)")
            ax.set_ylabel("Percent Alive")
            ax.set_title("Gene Expression Kaplan Meier Survival Curve")
            ax.set_ylim((0, 1))

            plt.show()
        elif make_plot == 'KM miRNA':
            surv2 = surv.copy(True)
            surv2 = surv2.loc[surv2["case_id"].isin(test_labels.values)]
            surv2 = surv2.set_index("case_id")
            surv2 = surv2.reindex(test_class.index.values)
            prd = clf_miRNA.predict(miRNA_test.loc[:, fea_miRNA])
            surv2["label_new"] = prd

            count = 0
            preds = clf_miRNA.predict(miRNA_test.loc[:, fea_miRNA])
            for i in range(test_class.shape[0]):
                if preds[i] == test_class.iloc[i, 0]:
                    count += 1
            print(count)

            T2 = surv2["days_to_death"]
            E2 = surv2["vital_status"]
            class02 = (surv2["label_new"] == 0)
            kmf.fit(T2[class02], event_observed=E2[class02], label="Low Survival")
            kmf.plot_survival_function(ax=ax, ci_show=False)
            kmf.fit(T2[~class02], event_observed=E2[~class02], label="High Survival")
            kmf.plot_survival_function(ax=ax, ci_show=False)
            ax.set_xlabel("Duration (days)")
            ax.set_ylabel("Percent Alive")
            ax.set_title("miRNA Expression Kaplan Meier Survival Curve")
            ax.set_ylim((0, 1))

            plt.show()
        elif make_plot == 'KM Meth':
            surv2 = surv.copy(True)
            surv2 = surv2.loc[surv2["case_id"].isin(test_labels.values)]
            surv2 = surv2.set_index("case_id")
            surv2 = surv2.reindex(test_class.index.values)
            prd = clf_meth.predict(meth_test.loc[:, fea_meth])
            surv2["label_new"] = prd

            count = 0
            preds = clf_meth.predict(meth_test.loc[:, fea_meth])
            for i in range(test_class.shape[0]):
                if preds[i] == test_class.iloc[i, 0]:
                    count += 1
            print(count)

            T2 = surv2["days_to_death"]
            E2 = surv2["vital_status"]
            class02 = (surv2["label_new"] == 0)
            kmf.fit(T2[class02], event_observed=E2[class02], label="Low Survival")
            kmf.plot_survival_function(ax=ax, ci_show=False)
            kmf.fit(T2[~class02], event_observed=E2[~class02], label="High Survival")
            kmf.plot_survival_function(ax=ax, ci_show=False)
            ax.set_xlabel("Duration (days)")
            ax.set_ylabel("Percent Alive")
            ax.set_title("DNA Methylation Kaplan Meier Survival Curve")
            ax.set_ylim((0, 1))

            plt.show()
        elif make_plot == 'KM CNV':
            surv2 = surv.copy(True)
            surv2 = surv2.loc[surv2["case_id"].isin(test_labels.values)]
            surv2 = surv2.set_index("case_id")
            surv2 = surv2.reindex(test_class.index.values)
            prd = clf_CNV.predict(CNV_test.loc[:, fea_CNV])
            surv2["label_new"] = prd

            count = 0
            preds = clf_CNV.predict(CNV_test.loc[:, fea_CNV])
            for i in range(test_class.shape[0]):
                if preds[i] == test_class.iloc[i, 0]:
                    count += 1
            print(count)

            T2 = surv2["days_to_death"]
            E2 = surv2["vital_status"]
            class02 = (surv2["label_new"] == 0)
            kmf.fit(T2[class02], event_observed=E2[class02], label="Low Survival")
            kmf.plot_survival_function(ax=ax, ci_show=False)
            kmf.fit(T2[~class02], event_observed=E2[~class02], label="High Survival")
            kmf.plot_survival_function(ax=ax, ci_show=False)
            ax.set_xlabel("Duration (days)")
            ax.set_ylabel("Percent Alive")
            ax.set_title("CNV Kaplan Meier Survival Curve")
            ax.set_ylim((0, 1))

            plt.show()
    elif make_plot == 'Gene Distribution' or make_plot == 'miRNA Distribution' or make_plot == 'Meth Distribution' \
            or make_plot == 'CNV Distribution':
        if make_plot == 'Gene Distribution':
            lbl = lbl.set_index("case_id")
            boxframe = gene.loc[fea_gene, :].T
            boxframe = boxframe.iloc[:, 0:10]
            cols = boxframe.columns
            g = genes.loc[cols,:]
            boxframe = pd.concat([boxframe, lbl[["label"]]], axis=1)
            c1_gene = boxframe[boxframe["label"] == 1]
            c1_gene = c1_gene.iloc[:, 0:10]
            c2_gene = boxframe[boxframe["label"] == 0]
            c2_gene = c2_gene.iloc[:, 0:10]
            for i in range(0,10):
                cl = ['High', 'Low']
                ind = [1, 2]
                plt.subplot(2, 5, i + 1)
                p1 = c1_gene.iloc[:, i]
                p2 = c2_gene.iloc[:, i]
                genes_to_plot = (p1, p2)
                plt.boxplot(genes_to_plot)
                plt.ylabel('Gene Expression (FPKM)', fontsize=16)
                if i > 4:
                    plt.xlabel('Survival Class', fontsize=16)
                plt.xticks(ind, cl, fontsize=16)
                max = 0
                max1 = (p1.values.max())
                max2 = (p2.values.max())
                if max1 > max2:
                    max = max1
                else:
                    max = max2
                plt.yticks([0, max], rotation='vertical')
                # print(cols[i])
                # title_str = 'Survival Breakdown of CNV Features by Gain/Loss: {}'
                # title_str = title_str.format(cols[i])
                plt.title(genes.loc[cols[i], 'hgnc_symbol'], fontsize=18)
            plt.suptitle('Top 10 Gene Feature Distributions by Survival Class', fontsize=22)
            plt.show()
        elif make_plot == 'miRNA Distribution':
            lbl = lbl.set_index("case_id")
            boxframe = miRNA.loc[fea_miRNA, :].T
            boxframe = boxframe.iloc[:, 0:10]
            cols = boxframe.columns
            boxframe = pd.concat([boxframe, lbl[["label"]]], axis=1)
            c1_miRNA = boxframe[boxframe["label"] == 1]
            c1_miRNA = c1_miRNA.iloc[:, 0:10]
            c2_miRNA = boxframe[boxframe["label"] == 0]
            c2_miRNA = c2_miRNA.iloc[:, 0:10]
            for i in range(0, 10):
                cl = ['High', 'Low']
                ind = [1, 2]
                plt.subplot(2, 5, i + 1)
                p1 = c1_miRNA.iloc[:, i]
                p2 = c2_miRNA.iloc[:, i]
                genes_to_plot = (p1, p2)
                plt.boxplot(genes_to_plot)
                plt.ylabel('miRNA Expression (RPM)', fontsize=16)
                if i > 4:
                    plt.xlabel('Survival Class', fontsize=16)
                plt.xticks(ind, cl, fontsize=16)
                max = 0
                max1 = (p1.values.max())
                max2 = (p2.values.max())
                if max1 > max2:
                    max = max1
                else:
                    max = max2
                plt.yticks([0, max], rotation='vertical')
                plt.title(cols[i], fontsize=18)
            plt.suptitle('Top 10 miRNA Feature Distributions by Survival Class', fontsize=22)
            plt.show()
        elif make_plot == 'Meth Distribution':
            lbl = lbl.set_index("case_id")
            boxframe = meth.loc[fea_meth, :].T
            boxframe = boxframe.iloc[:, 0:10]
            cols = boxframe.columns
            boxframe = pd.concat([boxframe, lbl[["label"]]], axis=1)
            c1_meth = boxframe[boxframe["label"] == 1]
            c1_mmeth = c1_meth.iloc[:, 0:10]
            c2_meth = boxframe[boxframe["label"] == 0]
            c2_meth = c2_meth.iloc[:, 0:10]
            for i in range(0, 10):
                cl = ['High', 'Low']
                ind = [1, 2]
                plt.subplot(2, 5, i + 1)
                p1 = c1_meth.iloc[:, i]
                p2 = c2_meth.iloc[:, i]
                genes_to_plot = (p1, p2)
                plt.boxplot(genes_to_plot)
                plt.ylabel('DNA Methylation (Beta Value)', fontsize=16)
                if i > 4:
                    plt.xlabel('Survival Class', fontsize=16)
                plt.xticks(ind, cl, fontsize=16)
                max = 0
                max1 = (p1.values.max())
                max2 = (p2.values.max())
                if max1 > max2:
                    max = max1
                else:
                    max = max2
                plt.yticks([0, max], rotation='vertical')
                plt.title(cols[i], fontsize=18)
            plt.suptitle('Top 10 DNA Methylation Feature Distributions by Survival Class', fontsize=22)
            plt.show()
        elif make_plot == 'CNV Distribution':
            lbl = lbl.set_index("case_id")
            cnv_lbl = pd.concat([CNV.loc[fea_CNV, :].T, lbl["label"]], axis=1)
            c1_CNV = cnv_lbl[cnv_lbl["label"] == 1]
            c1_CNV = c1_CNV.iloc[:, 0:16]
            cols = c1_CNV.columns.values
            c2_CNV = cnv_lbl[cnv_lbl["label"] == 0]
            c2_CNV = c2_CNV.iloc[:, 0:16]
            for i in range(0, 10):
                c1_CNV_1 = c1_CNV.iloc[:, i][c1_CNV.iloc[:, i] == 1].sum().sum()
                c1_CNV_neg1 = -1 * (c1_CNV.iloc[:, i][c1_CNV.iloc[:, i] == -1].sum().sum())
                c1_CNV_0 = (247 * 1) - (c1_CNV_1 + c1_CNV_neg1)
                c1_tot = (247 * 1)
                c1_pct_1 = c1_CNV_1 / c1_tot
                c1_pct_neg1 = c1_CNV_neg1 / c1_tot
                c1_pct_0 = c1_CNV_0 / c1_tot

                c2_CNV_1 = c2_CNV.iloc[:, i][c2_CNV.iloc[:, i] == 1].sum().sum()
                c2_CNV_neg1 = -1 * (c2_CNV.iloc[:, i][c2_CNV.iloc[:, i] == -1].sum().sum())
                c2_CNV_0 = (95 * 1) - (c2_CNV_1 + c2_CNV_neg1)
                c2_tot = (95 * 1)
                c2_pct_1 = c2_CNV_1 / c2_tot
                c2_pct_neg1 = c2_CNV_neg1 / c2_tot
                c2_pct_0 = c2_CNV_0 / c2_tot
                ones = (c1_pct_1, c2_pct_1)
                negones = (c1_pct_neg1, c2_pct_neg1)
                zeros = (c1_pct_0, c2_pct_0)
                hold = tuple(map(operator.add, zeros, negones))

                cl = ['High', 'Low']
                ind = [1, 2]
                width = 0.5

                plt.subplot(2, 5, i + 1)
                p1 = plt.bar(ind, negones, width)
                p2 = plt.bar(ind, zeros, width, bottom=negones)
                p3 = plt.bar(ind, ones, width, bottom=hold)
                if i == 0 or i == 5:
                    plt.ylabel('Percent of Top Features', fontsize=16)
                plt.xticks(ind, cl, fontsize=16)
                if i > 4:
                    plt.xlabel('Survival Class', fontsize=16)
                plt.yticks(np.arange(0, 1, 0.1))
                plt.legend((p1[0], p2[0], p3[0]), ('-1', '0', '1'))
                plt.title(genes.loc[cols[i][0:15], 'hgnc_symbol'], fontsize=18)
            plt.suptitle('Top 10 CNV Feature Distributions by Survival Class', fontsize=22)
            plt.show()
    elif make_plot == 'Heat Map Gene' or make_plot == 'Heat Map miRNA' or make_plot == 'Heat Map Meth' \
        or make_plot == 'Heat Map CNV':
        if make_plot == 'Heat Map Gene':
            clf_gene_h, fea_gene_h, heat_gene = do_cv(gene_train, train_class, gene_test, test_class, 'ttest', 'gene', 100,
                                                  2)
            n_c = 27
            n_feat = 49
            up_bound = 100
            lin_kern = np.empty([n_c, n_feat])  # c vals ( see in cv_2 ), # feats (n_max - 2 / 2)
            poly_kern = np.empty([n_c, n_feat])
            rbf_kern = np.empty([n_c, n_feat])
            sig_kern = np.empty([n_c, n_feat])

            linvec = []
            polvec = []
            rbfvec = []
            sigvec = []
            heat_gene_out = heat_gene[1:len(heat_gene), :]
            for n, i in enumerate(heat_gene_out):
                if n % 4 == 0:
                    linvec.append(i[3])
                elif n % 4 == 1:
                    polvec.append(i[3])
                elif n % 4 == 2:
                    rbfvec.append(i[3])
                elif n % 4 == 3:
                    sigvec.append(i[3])
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    lin_kern[c][n] = linvec[c + (27 * n)]
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    poly_kern[c][n] = polvec[c + (27 * n)]
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    rbf_kern[c][n] = rbfvec[c + (27 * n)]
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    sig_kern[c][n] = sigvec[c + (27 * n)]
            ax = plt.subplot(111)
            chosen = []
            out_kern = None
            if clf_gene_h.get_params(False)['kernel'] == 'linear':
                im = plt.imshow(lin_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = linvec
                out_kern = lin_kern.copy(True)
            elif clf_gene_h.get_params(False)['kernel'] == 'rbf':
                im = plt.imshow(rbf_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = rbfvec
                out_kern = rbf_kern.copy(True)
            elif clf_gene_h.get_params(False)['kernel'] == 'sigmoid':
                im = plt.imshow(sig_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = sigvec
                out_kern = sig_kern.copy(True)
            elif clf_gene_h.get_params(False)['kernel'] == 'poly':
                im = plt.imshow(poly_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = polvec
                out_kern = poly_kern.copy(True)
            plt.colorbar()
            plt.clim(min(chosen), max(chosen))

            ax.set_xticks(np.arange(n_feat))
            ax.set_yticks(np.arange(n_c))
            ax.set_xticklabels(range(2, up_bound, 2), fontsize=12)
            ax.set_yticklabels(
                [.001, .005, 0.1, .5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 50, 75, 100, 150,
                 200, 250], fontsize=14)
            ax.set_xlabel("Number of features", fontsize=20)
            ax.set_ylabel("C value", fontsize=20)
            plt.setp(ax.get_xticklabels(), rotation=90, rotation_mode="anchor")
            for j in range(0, n_c):
                for k in range(0, n_feat):
                    text = ax.text(k, j, '%.2f' % out_kern[j, k],
                                   ha="center", va="center", color="w", fontsize=6)
            ax.set_title(
                "Gene Expression Parameter Sensitivity (Accuracy) - %s kernel" % clf_gene_h.get_params(False)['kernel'],
                fontsize=24)
            plt.show()
        elif make_plot == 'Heat Map miRNA':
            clf_miRNA_h, fea_miRNA_h, heat_miRNA = do_cv(miRNA_train, train_class, miRNA_test, test_class, 'minfo', 'miRNA', 100,
                                                  2)
            n_c = 27
            n_feat = 49
            up_bound = 100
            lin_kern = np.empty([n_c, n_feat])  # c vals ( see in cv_2 ), # feats (n_max - 2 / 2)
            poly_kern = np.empty([n_c, n_feat])
            rbf_kern = np.empty([n_c, n_feat])
            sig_kern = np.empty([n_c, n_feat])

            linvec = []
            polvec = []
            rbfvec = []
            sigvec = []
            heat_miRNA_out = heat_miRNA[1:len(heat_miRNA), :]
            for n, i in enumerate(heat_miRNA_out):
                if n % 4 == 0:
                    linvec.append(i[3])
                elif n % 4 == 1:
                    polvec.append(i[3])
                elif n % 4 == 2:
                    rbfvec.append(i[3])
                elif n % 4 == 3:
                    sigvec.append(i[3])
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    lin_kern[c][n] = linvec[c + (27 * n)]
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    poly_kern[c][n] = polvec[c + (27 * n)]
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    rbf_kern[c][n] = rbfvec[c + (27 * n)]
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    sig_kern[c][n] = sigvec[c + (27 * n)]
            ax = plt.subplot(111)
            chosen = []
            out_kern = None
            if clf_miRNA_h.get_params(False)['kernel'] == 'linear':
                im = plt.imshow(lin_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = linvec
                out_kern = lin_kern.copy(True)
            elif clf_miRNA_h.get_params(False)['kernel'] == 'rbf':
                im = plt.imshow(rbf_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = rbfvec
                out_kern = rbf_kern.copy(True)
            elif clf_miRNA_h.get_params(False)['kernel'] == 'sigmoid':
                im = plt.imshow(sig_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = sigvec
                out_kern = sig_kern.copy(True)
            elif clf_miRNA_h.get_params(False)['kernel'] == 'poly':
                im = plt.imshow(poly_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = polvec
                out_kern = poly_kern.copy(True)
            plt.colorbar()
            plt.clim(min(chosen), max(chosen))

            ax.set_xticks(np.arange(n_feat))
            ax.set_yticks(np.arange(n_c))
            ax.set_xticklabels(range(2, up_bound, 2), fontsize=12)
            ax.set_yticklabels(
                [.001, .005, 0.1, .5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 50, 75, 100, 150,
                 200, 250], fontsize=14)
            ax.set_xlabel("Number of features", fontsize=20)
            ax.set_ylabel("C value", fontsize=20)
            plt.setp(ax.get_xticklabels(), rotation=90, rotation_mode="anchor")
            for j in range(0, n_c):
                for k in range(0, n_feat):
                    text = ax.text(k, j, '%.2f' % out_kern[j, k],
                                   ha="center", va="center", color="w", fontsize=6)
            ax.set_title(
                "miRNA Expression Parameter Sensitivity (Accuracy) - %s kernel" % clf_miRNA_h.get_params(False)['kernel'],
                fontsize=24)
            plt.show()
        elif make_plot == 'Heat Map Meth':
            clf_meth_h, fea_meth_h, heat_meth = do_cv(meth_train, train_class, meth_test, test_class, 'minfo', 'meth', 100,
                                                  2)
            n_c = 27
            n_feat = 49
            up_bound = 100
            lin_kern = np.empty([n_c, n_feat])  # c vals ( see in cv_2 ), # feats (n_max - 2 / 2)
            poly_kern = np.empty([n_c, n_feat])
            rbf_kern = np.empty([n_c, n_feat])
            sig_kern = np.empty([n_c, n_feat])

            linvec = []
            polvec = []
            rbfvec = []
            sigvec = []
            heat_meth_out = heat_meth[1:len(heat_meth), :]
            for n, i in enumerate(heat_meth_out):
                if n % 4 == 0:
                    linvec.append(i[3])
                elif n % 4 == 1:
                    polvec.append(i[3])
                elif n % 4 == 2:
                    rbfvec.append(i[3])
                elif n % 4 == 3:
                    sigvec.append(i[3])
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    lin_kern[c][n] = linvec[c + (27 * n)]
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    poly_kern[c][n] = polvec[c + (27 * n)]
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    rbf_kern[c][n] = rbfvec[c + (27 * n)]
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    sig_kern[c][n] = sigvec[c + (27 * n)]
            ax = plt.subplot(111)
            chosen = []
            out_kern = None
            if clf_meth_h.get_params(False)['kernel'] == 'linear':
                im = plt.imshow(lin_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = linvec
                out_kern = lin_kern.copy(True)
            elif clf_meth_h.get_params(False)['kernel'] == 'rbf':
                im = plt.imshow(rbf_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = rbfvec
                out_kern = rbf_kern.copy(True)
            elif clf_meth_h.get_params(False)['kernel'] == 'sigmoid':
                im = plt.imshow(sig_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = sigvec
                out_kern = sig_kern.copy(True)
            elif clf_meth_h.get_params(False)['kernel'] == 'poly':
                im = plt.imshow(poly_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = polvec
                out_kern = poly_kern.copy(True)
            plt.colorbar()
            plt.clim(min(chosen), max(chosen))

            ax.set_xticks(np.arange(n_feat))
            ax.set_yticks(np.arange(n_c))
            ax.set_xticklabels(range(2, up_bound, 2), fontsize=12)
            ax.set_yticklabels(
                [.001, .005, 0.1, .5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 50, 75, 100, 150,
                 200, 250], fontsize=14)
            ax.set_xlabel("Number of features", fontsize=20)
            ax.set_ylabel("C value", fontsize=20)
            plt.setp(ax.get_xticklabels(), rotation=90, rotation_mode="anchor")
            for j in range(0, n_c):
                for k in range(0, n_feat):
                    text = ax.text(k, j, '%.2f' % out_kern[j, k],
                                   ha="center", va="center", color="w", fontsize=6)
            ax.set_title(
                "DNA Methylation Parameter Sensitivity (Accuracy) - %s kernel" % clf_meth_h.get_params(False)['kernel'],
                fontsize=24)
            plt.show()
        elif make_plot == 'Heat Map CNV':
            clf_CNV_h, fea_CNV_h, heat_CNV = do_cv(CNV_train, train_class, CNV_test, test_class, 'minfo', 'CNV', 100,
                                                  2)
            n_c = 27
            n_feat = 49
            up_bound = 100
            lin_kern = np.empty([n_c, n_feat])  # c vals ( see in cv_2 ), # feats (n_max - 2 / 2)
            poly_kern = np.empty([n_c, n_feat])
            rbf_kern = np.empty([n_c, n_feat])
            sig_kern = np.empty([n_c, n_feat])

            linvec = []
            polvec = []
            rbfvec = []
            sigvec = []
            heat_CNV_out = heat_CNV[1:len(heat_CNV), :]
            for n, i in enumerate(heat_CNV_out):
                if n % 4 == 0:
                    linvec.append(i[3])
                elif n % 4 == 1:
                    polvec.append(i[3])
                elif n % 4 == 2:
                    rbfvec.append(i[3])
                elif n % 4 == 3:
                    sigvec.append(i[3])
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    lin_kern[c][n] = linvec[c + (27 * n)]
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    poly_kern[c][n] = polvec[c + (27 * n)]
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    rbf_kern[c][n] = rbfvec[c + (27 * n)]
            for n in range(0, n_feat):
                for c in range(0, n_c):
                    sig_kern[c][n] = sigvec[c + (27 * n)]
            ax = plt.subplot(111)
            chosen = []
            out_kern = None
            if clf_CNV_h.get_params(False)['kernel'] == 'linear':
                im = plt.imshow(lin_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = linvec
                out_kern = lin_kern.copy(True)
            elif clf_CNV_h.get_params(False)['kernel'] == 'rbf':
                im = plt.imshow(rbf_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = rbfvec
                out_kern = rbf_kern.copy(True)
            elif clf_CNV_h.get_params(False)['kernel'] == 'sigmoid':
                im = plt.imshow(sig_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = sigvec
                out_kern = sig_kern.copy(True)
            elif clf_CNV_h.get_params(False)['kernel'] == 'poly':
                im = plt.imshow(poly_kern, cmap=plt.cm.get_cmap('coolwarm', 50))
                chosen = polvec
                out_kern = poly_kern.copy(True)
            plt.colorbar()
            plt.clim(min(chosen), max(chosen))

            ax.set_xticks(np.arange(n_feat))
            ax.set_yticks(np.arange(n_c))
            ax.set_xticklabels(range(2, up_bound, 2), fontsize=12)
            ax.set_yticklabels(
                [.001, .005, 0.1, .5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 50, 75, 100, 150,
                 200, 250], fontsize=14)
            ax.set_xlabel("Number of features", fontsize=20)
            ax.set_ylabel("C value", fontsize=20)
            plt.setp(ax.get_xticklabels(), rotation=90, rotation_mode="anchor")
            for j in range(0, n_c):
                for k in range(0, n_feat):
                    text = ax.text(k, j, '%.2f' % out_kern[j, k],
                                   ha="center", va="center", color="w", fontsize=6)
            ax.set_title(
                "Copy Number Variation Parameter Sensitivity (Accuracy) - %s kernel" % clf_CNV_h.get_params(False)['kernel'],
                fontsize=24)
            plt.show()
    elif make_plot == None:
        pass

    if compute == 'custom':
        return clf_gene, fea_gene, clf_miRNA, fea_miRNA, clf_meth, fea_meth, clf_CNV, fea_CNV, clf

    return tr_score,te_score

# print(pipeline(True,'precomputed','barplot',42,"C:\\Users\\jonat\\Documents\\Spring 2019 Classes\\4813\\BHI_data"))






