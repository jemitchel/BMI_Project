import feat_select as fs
# import pipeline_gui as pp
import pipeline as pp
import test_pt as tp
import test_pt_file as tpf
import load as ld
import pandas as pd
#import train_ind as ti
#import train_comb as tc
import sys
import os
import tkinter as tk
from tkinter import filedialog
import random
import webbrowser


class GUI:

    def __init__(self, master, dir):

        self.master = master
        self.gene_feat = []
        self.miRNA_feat = []
        self.meth_feat = []
        self.CNV_feat = []

        master.title("Multi Modal Genomics - Breast Cancer Survival Prediction")

        frame1 = tk.Frame(master, bd=0, highlightbackground="black", highlightcolor="black",
                          highlightthickness=1, pady=5)
        frame1.pack(side="top", anchor=tk.NW)
        frame2 = tk.Frame(master, bd=0, highlightbackground="black", highlightcolor="black",
                          highlightthickness=1, pady=5)
        frame2.pack(side="bottom")

        label1 = tk.Label(frame1, text="Optional inputs for integrated classification:", background="white")
        label1.pack(anchor=tk.NW)

        frameModalities = tk.Frame(frame1)
        frameModalities.pack(anchor=tk.N,side="left",padx=20)
        labelModalities = tk.Label(frameModalities,
                                   text="Select which method to use for\nfeature selection of  variables:")
        labelModalities.pack(anchor=tk.W)

        var_t = tk.IntVar()
        checkbox_t = tk.Checkbutton(frameModalities, text="T test (cont) / Chi-square (disc)", variable=var_t)
        checkbox_t.pack(anchor=tk.W)
        var_mRMR = tk.IntVar()
        checkbox_mRMR = tk.Checkbutton(frameModalities, text="mRMR", variable=var_mRMR)
        checkbox_mRMR.pack(anchor=tk.W)
        var_mInfo = tk.IntVar()
        checkbox_mInfo = tk.Checkbutton(frameModalities, text="Mutual Information", variable=var_mInfo)
        checkbox_mInfo.pack(anchor=tk.W)

        frameRem = tk.Frame(frame1)
        frameRem.pack(anchor=tk.N, side="left", padx=20)
        labelRem = tk.Label(frameRem, text="Check ""remove"" below to remove non-coding\nRNA "
                                           "sequences from RNA data sets:")
        labelRem.pack(anchor=tk.N)

        var_rem = tk.IntVar()
        checkbox_rem = tk.Checkbutton(frameRem, text="Remove", variable=var_rem)
        checkbox_rem.pack(anchor=tk.W)


        frameRand = tk.Frame(frame1)
        frameRand.pack(anchor=tk.N, side="left", padx=20)
        labelRand = tk.Label(frameRand, text="Select a whole number to be used as a seed"
                                                  "\nfor random number generation:")
        labelRand.pack(anchor=tk.N)

        var_Rand = tk.IntVar()
        var_Rand = 5
        entry_Rand = tk.Entry(frameRand, textvariable=str(var_Rand))
        entry_Rand.insert("end", str(var_Rand))
        def rnd():
            var_Rand = random.randint(1,50)
            entry_Rand.delete(0,tk.END)
            entry_Rand.insert("end", str(var_Rand))
        button_Rand = tk.Button(frameRand, text="Random", foreground="black", bd=2,
                                     command=rnd, height=2, width=10)
        button_Rand.pack()

        entry_Rand.pack()

        frameTrain = tk.Frame(frame1)
        frameTrain.pack(anchor=tk.N, side="left", padx=20)
        labelTrain = tk.Label(frameTrain, text="Select percentage of total patients to be "
                                                     "\nincluded in the test group for classification:")
        labelTrain.pack(anchor=tk.N)

        var_Train = tk.StringVar()
        entry_Train = tk.Entry(frameTrain, textvariable=var_Train)
        entry_Train.insert("end", "0.15")
        entry_Train.pack()

        frame3 = tk.Frame(frame2, bd=0, highlightbackground="black", highlightcolor="black",
                          highlightthickness=1, pady=5)
        frame3.pack(side="left")
        frame4 = tk.Frame(frame2, bd=0, highlightbackground="black", highlightcolor="black",
                          highlightthickness=1, pady=5)
        frame4.pack(side="right")

        label2 = tk.Label(frame3, text="Classification Results:", background="white")
        label2.pack(side="top", anchor=tk.NW)
        frameOutputClassifier = tk.Frame(frame3)
        frameOutputClassifier.pack(anchor=tk.NW, side="top", padx=5, pady=5)
        labelOutputClassifier = tk.Label(frameOutputClassifier, text="Classifier Output:")
        labelOutputClassifier.pack(anchor=tk.NW)
        scrollbar_out = tk.Scrollbar(frameOutputClassifier)
        scrollbar_out.pack(side="right", fill=tk.Y)
        text_OutputClassifier = tk.Text(frameOutputClassifier, state=tk.DISABLED, height=5, width=80,
                                        yscrollcommand=scrollbar_out.set)
        text_OutputClassifier.pack(expand=True, fill='both')

        frameOutputFeatures = tk.Frame(frame3)
        frameOutputFeatures.pack(anchor=tk.SW, side="bottom", padx=5, pady=5)

        frameOutputGene = tk.Frame(frameOutputFeatures)
        frameOutputGene.pack(anchor=tk.SW, side="left", padx=5, pady=5)
        labelOutputGene = tk.Label(frameOutputGene, text="Selected Features:\nGene Expression")
        labelOutputGene.pack(side="top", anchor=tk.SW)

        scrollbar_outGene = tk.Scrollbar(frameOutputGene)
        scrollbar_outGene.pack(side="right", fill=tk.Y)
        text_OutputGene = tk.Text(frameOutputGene, state=tk.DISABLED, height=5, width=20,
                                      yscrollcommand=scrollbar_outGene.set)
        text_OutputGene.pack(expand=True, fill='both')

        def openGeneFeats():
            # windowGene = tk.Toplevel(root, height=800, width=800)
            windowGene = tk.Tk()
            windowGene.geometry('250x700')
            canv = tk.Canvas(windowGene, height=800, width=800)


            frame = tk.Frame(canv, bd=0, highlightbackground="black", highlightcolor="black",
                              highlightthickness=1, pady=5)
            frame.pack(side="top", anchor=tk.NW)
            hyperlink = tk.Label(frame, text="Gene database hyperlink", fg="blue", cursor="hand2")
            hyperlink.pack()
            def click(event):
                url = 'https://useast.ensembl.org/index.html'
                webbrowser.open_new(url)
            hyperlink.bind("<Button-1>", click)
            frame2 = tk.Frame(canv, bd=0, highlightbackground="black", highlightcolor="black",
                              highlightthickness=1, pady=5,height=800)
            frame2.pack(side="bottom")

            #for n in num features, make a text entry with hyperlink pointing to ensembl id
            #https://useast.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g=%s;r=X:77910739-78129295
            #https: // www.proteinatlas.org / %s / tissue
            lbl_dict = {}
            url_dict = {}
            for i in self.gene_feat:
                url_dict[i] = 'https://useast.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g=%s;r=X:77910739-78129295' % i
                action = lambda x = i: webbrowser.open_new(url_dict[x])

                lbl_dict[i] = tk.Button(frame2, text=i,
                               fg="blue", cursor="hand2", command=action)
                lbl_dict[i].pack()
            genescroll = tk.Scrollbar(canv)

            canv.create_window((0, 0), anchor='nw', window=frame, height=35, state='normal')
            canv.create_window((0, 36), anchor='nw', window=frame2, height=800, state='normal')
            canv.update_idletasks()
            canv.configure(scrollregion=canv.bbox('all'),
                             yscrollcommand=genescroll.set)
            genescroll.config(command=canv.yview)
            canv.pack(fill='both', expand=True, side='left')
            genescroll.pack(side="right", fill=tk.Y)




        button_OutputGene = tk.Button(frameOutputGene, text="Gene Features", foreground="black", bd=2,
                                     command=openGeneFeats, height=1, width=18)
        button_OutputGene.pack()

        frameOutputmi = tk.Frame(frameOutputFeatures)
        frameOutputmi.pack(anchor=tk.SW, side="left", padx=5, pady=5)
        labelOutputmi = tk.Label(frameOutputmi, text="Selected Features:\nmiRNA Expression")
        labelOutputmi.pack(side="top", anchor=tk.SW)

        scrollbar_outmi = tk.Scrollbar(frameOutputmi)
        scrollbar_outmi.pack(side="right", fill=tk.Y)
        text_Outputmi = tk.Text(frameOutputmi, state=tk.DISABLED, height=5, width=20,
                                  yscrollcommand=scrollbar_outmi.set)
        text_Outputmi.pack(expand=True, fill='both')
        def openmiFeats():
            windowmi = tk.Tk()
            windowmi.geometry('250x700')
            canv = tk.Canvas(windowmi, height=800, width=800)

            frame = tk.Frame(canv, bd=0, highlightbackground="black", highlightcolor="black",
                             highlightthickness=1, pady=5)
            frame.pack(side="top", anchor=tk.NW)
            hyperlink = tk.Label(frame, text="miRNA database hyperlink", fg="blue", cursor="hand2")
            hyperlink.pack()

            def click(event):
                url = 'http://www.mirbase.org/'
                webbrowser.open_new(url)

            hyperlink.bind("<Button-1>", click)
            frame2 = tk.Frame(canv, bd=0, highlightbackground="black", highlightcolor="black",
                              highlightthickness=1, pady=5, height=800)
            frame2.pack(side="bottom")

            lbl_dict = {}
            url_dict = {}
            for i in self.miRNA_feat:
                url_dict[
                    i] = 'http://www.mirbase.org/textsearch.shtml?q=%s' % i
                action = lambda x=i: webbrowser.open_new(url_dict[x])

                lbl_dict[i] = tk.Button(frame2, text=i,
                                        fg="blue", cursor="hand2", command=action)
                lbl_dict[i].pack()
            miscroll = tk.Scrollbar(canv)

            canv.create_window((0, 0), anchor='nw', window=frame, height=35, state='normal')
            canv.create_window((0, 36), anchor='nw', window=frame2, height=800, state='normal')
            canv.update_idletasks()
            canv.configure(scrollregion=canv.bbox('all'),
                           yscrollcommand=miscroll.set)
            miscroll.config(command=canv.yview)
            canv.pack(fill='both', expand=True, side='left')
            miscroll.pack(side="right", fill=tk.Y)


        button_Outputmi = tk.Button(frameOutputmi, text="miRNA Features", foreground="black", bd=2,
                                     command=openmiFeats, height=1, width=18)
        button_Outputmi.pack()

        frameOutputMeth = tk.Frame(frameOutputFeatures)
        frameOutputMeth.pack(anchor=tk.SW, side="left", padx=5, pady=5)
        labelOutputMeth = tk.Label(frameOutputMeth, text="Selected Features:\nDNA Methylation")
        labelOutputMeth.pack(side="top", anchor=tk.SW)

        scrollbar_outMeth = tk.Scrollbar(frameOutputMeth)
        scrollbar_outMeth.pack(side="right", fill=tk.Y)
        text_OutputMeth = tk.Text(frameOutputMeth, state=tk.DISABLED, height=5, width=20,
                                yscrollcommand=scrollbar_outMeth.set)
        text_OutputMeth.pack(expand=True, fill='both')

        def openMethFeats():
            windowmeth = tk.Tk()
            windowmeth.geometry('250x700')
            canv = tk.Canvas(windowmeth, height=800, width=800)

            frame = tk.Frame(canv, bd=0, highlightbackground="black", highlightcolor="black",
                             highlightthickness=1, pady=5)
            frame.pack(side="top", anchor=tk.NW)
            hyperlink = tk.Label(frame, text="DNA Methylation database hyperlink", fg="blue", cursor="hand2")
            hyperlink.pack()

            def click(event):
                url = 'http://imethyl.iwate-megabank.org/'
                webbrowser.open_new(url)

            hyperlink.bind("<Button-1>", click)
            frame2 = tk.Frame(canv, bd=0, highlightbackground="black", highlightcolor="black",
                              highlightthickness=1, pady=5, height=800)
            frame2.pack(side="bottom")

            lbl_dict = {}
            url_dict = {}
            for i in self.meth_feat:
                url_dict[
                    i] = 'http://imethyl.iwate-megabank.org/jbrowse/?loc=%s&tracks=hg19,' \
                         'gencode_v19,gencode_v19_trs,CpGIslandsExt,HM450,RepeatMasker,IMM_CpG_CD4T,IMM_CpG_CD4T_avg,' \
                         'IMM_CpG_CD4T_sd,IMM_CpG_Mono,IMM_CpG_Mono_avg,IMM_CpG_Mono_sd,IMM_FPKM_CD4T,IMM_FPKM_Mono,IMM_SNV' % i
                action = lambda x=i: webbrowser.open_new(url_dict[x])

                lbl_dict[i] = tk.Button(frame2, text=i,
                                        fg="blue", cursor="hand2", command=action)
                lbl_dict[i].pack()
            methscroll = tk.Scrollbar(canv)

            canv.create_window((0, 0), anchor='nw', window=frame, height=35, state='normal')
            canv.create_window((0, 36), anchor='nw', window=frame2, height=800, state='normal')
            canv.update_idletasks()
            canv.configure(scrollregion=canv.bbox('all'),
                           yscrollcommand=methscroll.set)
            methscroll.config(command=canv.yview)
            canv.pack(fill='both', expand=True, side='left')
            methscroll.pack(side="right", fill=tk.Y)
            # http://imethyl.iwate-megabank.org/jbrowse/?loc=cg26833602&tracks=hg19,gencode_v19,gencode_v19_trs,CpGIslandsExt,HM450,RepeatMasker,IMM_CpG_CD4T,IMM_CpG_CD4T_avg,IMM_CpG_CD4T_sd,IMM_CpG_Mono,IMM_CpG_Mono_avg,IMM_CpG_Mono_sd,IMM_FPKM_CD4T,IMM_FPKM_Mono,IMM_SNV

        button_OutputMeth = tk.Button(frameOutputMeth, text="DNA Methylation Features", foreground="black", bd=2,
                                    command=openMethFeats, height=1, width=20)
        button_OutputMeth.pack()

        frameOutputCNV = tk.Frame(frameOutputFeatures)
        frameOutputCNV.pack(anchor=tk.SW, side="left", padx=5, pady=5)
        labelOutputCNV = tk.Label(frameOutputCNV, text="Selected Features:\nCopy Number Variation")
        labelOutputCNV.pack(side="top", anchor=tk.SW)

        scrollbar_outCNV = tk.Scrollbar(frameOutputCNV)
        scrollbar_outCNV.pack(side="right", fill=tk.Y)
        text_OutputCNV = tk.Text(frameOutputCNV, state=tk.DISABLED, height=5, width=20,
                                yscrollcommand=scrollbar_outCNV.set)
        text_OutputCNV.pack(expand=True, fill='both')

        def openCNVFeats():
            windowGene = tk.Tk()
            windowGene.geometry('250x700')
            canv = tk.Canvas(windowGene, height=800, width=800)

            frame = tk.Frame(canv, bd=0, highlightbackground="black", highlightcolor="black",
                             highlightthickness=1, pady=5)
            frame.pack(side="top", anchor=tk.NW)
            hyperlink = tk.Label(frame, text="Gene database hyperlink", fg="blue", cursor="hand2")
            hyperlink.pack()

            def click(event):
                url = 'https://useast.ensembl.org/index.html'
                webbrowser.open_new(url)

            hyperlink.bind("<Button-1>", click)
            frame2 = tk.Frame(canv, bd=0, highlightbackground="black", highlightcolor="black",
                              highlightthickness=1, pady=5, height=800)
            frame2.pack(side="bottom")

            # for n in num features, make a text entry with hyperlink pointing to ensembl id
            # https://useast.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g=%s;r=X:77910739-78129295
            # https: // www.proteinatlas.org / %s / tissue
            lbl_dict = {}
            url_dict = {}
            for i in self.gene_feat:
                url_dict[
                    i] = 'https://useast.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g=%s;r=X:77910739-78129295' % i[0:15]
                action = lambda x=i: webbrowser.open_new(url_dict[x])

                lbl_dict[i] = tk.Button(frame2, text=i,
                                        fg="blue", cursor="hand2", command=action)
                lbl_dict[i].pack()
            genescroll = tk.Scrollbar(canv)

            canv.create_window((0, 0), anchor='nw', window=frame, height=35, state='normal')
            canv.create_window((0, 36), anchor='nw', window=frame2, height=800, state='normal')
            canv.update_idletasks()
            canv.configure(scrollregion=canv.bbox('all'),
                           yscrollcommand=genescroll.set)
            genescroll.config(command=canv.yview)
            canv.pack(fill='both', expand=True, side='left')
            genescroll.pack(side="right", fill=tk.Y)

        button_OutputCNV = tk.Button(frameOutputCNV, text="CNV Features", foreground="black", bd=2,
                                    command=openCNVFeats, height=1, width=18)
        button_OutputCNV.pack()

        def runClassification():
            oneCheck = sum([var_t.get(), var_mRMR.get(), var_mInfo.get()])
            featSelectMethod = ''
            if oneCheck > 1:
                print("Only select one method for feature selection")
                sys.exit()
            if oneCheck == 1:
                if var_t.get() == 1:
                    featSelectMethod = 'ttest'
                elif var_mRMR.get() == 1:
                    featSelectMethod = 'mrmr'
                else:
                    featSelectMethod = 'minfo'
            else:
                print("Please select a method for feature selection")
                sys.exit()
            remove_zeros = False
            if var_rem == 1:
                remove_zeros = True

            clf_gene, fea_gene, clf_miRNA, fea_miRNA, clf_meth, fea_meth, clf_CNV, fea_CNV, clf = \
                pp.pipeline(remove_zeros, 'custom', None, var_Rand,
                            dir, method=featSelectMethod, test_size=float(var_Train.get()))
            #new func
            self.gene_feat = fea_gene
            self.miRNA_feat = fea_miRNA
            self.meth_feat = fea_meth
            self.CNV_feat = fea_CNV

            text_OutputClassifier.config(state="normal")
            text_OutputClassifier.insert("end", 'Gene Expression Classifier:\n %s \n' % clf_gene)
            text_OutputClassifier.insert("end", 'Micro RNA Expression Classifier:\n %s \n' % clf_miRNA)
            text_OutputClassifier.insert("end", 'DNA Methylation Classifier:\n %s \n' % clf_meth)
            text_OutputClassifier.insert("end", 'Copy Number Variation Classifier:\n %s \n'% clf_CNV)
            text_OutputClassifier.insert("end", 'Integrated Classifier: \n %s \n' % clf)
            text_OutputClassifier.config(state="disabled")
            text_OutputGene.config(state="normal")
            text_Outputmi.config(state="normal")
            text_OutputMeth.config(state="normal")
            text_OutputCNV.config(state="normal")
            for i in fea_gene:
                text_OutputGene.insert("end",'%s\n' % i)
            for i in fea_miRNA:
                text_Outputmi.insert("end",'%s\n' % i)
            for i in fea_meth:
                text_OutputMeth.insert("end", '%s\n' % i)
            for i in fea_CNV:
                text_OutputCNV.insert("end", '%s\n' % i)
            text_OutputGene.config(state="disabled")
            text_Outputmi.config(state="disabled")
            text_OutputMeth.config(state="disabled")
            text_OutputCNV.config(state="disabled")
            scrollbar_out.config(command=text_OutputClassifier.yview)
            scrollbar_outGene.config(command=text_OutputGene.yview)
            scrollbar_outmi.config(command=text_Outputmi.yview)
            scrollbar_outMeth.config(command=text_OutputMeth.yview)
            scrollbar_outCNV.config(command=text_OutputCNV.yview)



        button_run = tk.Button(frame1, text="RUN", foreground="white", bg="green", bd=2,
                                     command=runClassification, height=4, width=20)
        button_run.pack_propagate(0)
        button_run.pack(anchor=tk.N, side="top")

        def runOpt():
            clf_gene, fea_gene, clf_miRNA, fea_miRNA, clf_meth, fea_meth, clf_CNV, fea_CNV, clf = \
                ld.loadup(dir)

            self.gene_feat = fea_gene
            self.miRNA_feat = fea_miRNA
            self.meth_feat = fea_meth
            self.CNV_feat = fea_CNV

            text_OutputClassifier.config(state="normal")
            text_OutputClassifier.insert("end", 'Gene Expression Classifier:\n %s \n' % clf_gene)
            text_OutputClassifier.insert("end", 'Micro RNA Expression Classifier:\n %s \n' % clf_miRNA)
            text_OutputClassifier.insert("end", 'DNA Methylation Classifier:\n %s \n' % clf_meth)
            text_OutputClassifier.insert("end", 'Copy Number Variation Classifier:\n %s \n' % clf_CNV)
            text_OutputClassifier.insert("end", 'Integrated Classifier:\n %s \n' % clf)
            text_OutputClassifier.config(state="disabled")
            text_OutputGene.config(state="normal")
            text_Outputmi.config(state="normal")
            text_OutputMeth.config(state="normal")
            text_OutputCNV.config(state="normal")
            for i in fea_gene:
                text_OutputGene.insert("end", '%s\n' % i)
            for i in fea_miRNA:
                text_Outputmi.insert("end", '%s\n' % i)
            for i in fea_meth:
                text_OutputMeth.insert("end", '%s\n' % i)
            for i in fea_CNV:
                text_OutputCNV.insert("end", '%s\n' % i)
            text_OutputGene.config(state="disabled")
            text_Outputmi.config(state="disabled")
            text_OutputMeth.config(state="disabled")
            text_OutputCNV.config(state="disabled")
            scrollbar_out.config(command=text_OutputClassifier.yview)
            scrollbar_outGene.config(command=text_OutputGene.yview)
            scrollbar_outmi.config(command=text_Outputmi.yview)
            scrollbar_outMeth.config(command=text_OutputMeth.yview)
            scrollbar_outCNV.config(command=text_OutputCNV.yview)


        frame7 = tk.Frame(frame1)
        frame7.pack(anchor=tk.N, side='bottom')
        button_opt = tk.Button(frame7, text="Use Optimal Parameters", foreground="white", bg="green", bd=2,
                               command=runOpt, height=4, width=20)
        button_opt.pack_propagate(0)
        button_opt.pack(anchor=tk.N, side="left")

        frameTestPt = tk.Frame(frame4)
        frameTestPt.pack(anchor=tk.N, side="top", padx=20, pady=20)
        labelTestPt = tk.Label(frameTestPt, text="Select a patient to perform prediction on:")
        labelTestPt.pack(side="top", anchor=tk.NW)
        #

        def askOpenFile():
            file = filedialog.askopenfilename()
            if file != None:
                data = pd.read_csv(file)
                prediction, probability = tpf.test_pt_file(data)
                text_TestClass.config(state="normal")
                text_TestClass.insert("end", prediction)
                text_TestClass.config(state="disabled")
                text_TestProb.config(state="normal")
                text_TestProb.insert("end", probability)
                text_TestProb.config(state="disabled")

        buttonUploadPt = tk.Button(frameTestPt, text="Upload a patient data csv file", bg="yellow",
                                   command=askOpenFile)
        buttonUploadPt.pack(side="bottom")
        # Need to add what to do with file

        var_pt = tk.StringVar()
        txt_pt = tk.Entry(frameTestPt, textvariable=var_pt)
        txt_pt.pack(side="bottom")


        frameTestOutClass = tk.Frame(frame4)
        frameTestOutClass.pack(anchor=tk.N, side="left", padx=20, pady=20)
        labelTestOutClass = tk.Label(frameTestOutClass, text="Selected patient survival prediction: ")
        labelTestOutClass.pack(anchor=tk.NW)

        text_TestClass = tk.Text(frameTestOutClass, state=tk.DISABLED, height=5, width=15)
        text_TestClass.pack(expand=True, fill='both')  # check how to be able to show more

        frameTestOutProb = tk.Frame(frame4)
        frameTestOutProb.pack(anchor=tk.N, side="right", padx=20, pady=20)
        labelTestOutProb = tk.Label(frameTestOutProb, text="Survival prediction Probability: ")
        labelTestOutProb.pack(anchor=tk.NW)

        text_TestProb = tk.Text(frameTestOutProb, state=tk.DISABLED, height=5, width=15)
        text_TestProb.pack(expand=True, fill='both')  # check how to be able to show more

        def pt():
            patient = var_pt.get()
            print(patient)
            if patient == '':
                print("Must select a patient for prediction")
            else:
                prediction, probability = tp.test_pt(dir, patient)
                text_TestClass.config(state="normal")
                text_TestClass.insert("end",prediction)
                text_TestClass.config(state="disabled")
                text_TestProb.config(state="normal")
                text_TestProb.insert("end", probability)
                text_TestProb.config(state="disabled")

        btn_pt = tk.Button(frameTestPt, text="Predict", bg="green", command=pt)
        btn_pt.pack(side="right")

root = tk.Tk()
gui = GUI(root,"C:\\Users\\jonat\\Documents\\Spring 2019 Classes\\4813\\BHI_data")

root.mainloop()
