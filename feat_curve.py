from cv_2 import tr
import pandas as pd
from feat_select import select_features
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def make_feat_curve(X,y,X_test,y_test,f_sel,modality):
    max_n_feats = 20
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

    for i in range(2,max_n_feats,2):
        print(i)
        num_feats.append(i)
        X3 = pd.DataFrame(X2, copy=True)  # copies the original feature dataframe
        y3 = pd.DataFrame(y2, copy=True)  # copies the original feature dataframe
        clf,fea_ = tr(X3,y3,modality,f_sel,'none',feat_selected[0:i])

        clf.best_score_

        # stuff for test
        X4 = X2[fea_]
        X_test2 = X_test[fea_]
        train_acc.append(clf.score(X4, y2))
        test_acc.append(clf.score(X_test2, y_test))
        c1, c2, _ = roc_curve(y2.values.ravel(), clf.decision_function(X4).ravel())
        train_auc.append(auc(c1, c2))
        c1, c2, _ = roc_curve(y_test.values.ravel(), clf.decision_function(X_test2).ravel())
        test_auc.append(auc(c1, c2))

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.5, 0.8,.2], ylim=(0, 1))
    ax2 = fig.add_axes([0.1, 0.1, 0.8,.2], ylim=(0, 1))

    ax1.plot(num_feats,train_acc,'r')
    ax1.plot(num_feats,test_acc,'b')
    ax1.set_title('Train and Test Accuracy',fontsize=30)
    ax1.set_xlabel('Number of Features',fontsize=20)
    ax1.set_ylabel('Accuracy',fontsize=20)

    ax2.plot(num_feats,train_auc,'r')
    ax2.plot(num_feats,test_auc,'b')
    ax2.set_title('Train and Test AUC',fontsize=30)
    ax2.set_xlabel('Number of Features',fontsize=20)
    ax2.set_ylabel('AUC',fontsize=20)



    plt.show()

    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(num_feats, train_acc, 'r', num_feats, test_acc, 'b')
    # axs[0].set_title('Train and Test Accuracy')
    # # axs[0].set_xlabel('Number of Features')
    # axs[0].set_ylabel('Accuracy')
    # # fig.suptitle('This is a somewhat long figure title', fontsize=16)
    #
    # axs[1].plot(num_feats, train_auc, 'r', num_feats, test_auc, 'b')
    # axs[1].set_xlabel('Number of Features')
    # axs[1].set_title('Train and Test AUC')
    # axs[1].set_ylabel('AUC')
    #
    # axs[0] = fig.add_axes([0.1, 0.55, 0.8,.4],
    #                    xticklabels=[], ylim=(0, 1.2))
    # axs[1] = fig.add_axes([0.1, 0.1, 0.8,.4],
    #                    ylim=(0, 1.2))
    #
    # plt.show()


