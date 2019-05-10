import pandas as pd
import os

os.chdir("C:\\Users\\jonat\\Documents\\Spring 2019 Classes\\4813\\BHI_data")
labels = pd.read_csv('Label_selected.csv')

ndx = labels.index[labels['label'] > -1].tolist() #gets indices of patients to use
lbl = labels.iloc[ndx,[0,4]] #makes a new dataframe of patients to use (their IDs and survival response)

# miRNA dataset
miRNA = pd.read_csv('miRNA_selected.csv') #reads in dataset
miRNA = miRNA.set_index('miRNA_ID') #changes first column (miRNA_ID) to be indices of the dataframe
miRNA = miRNA[lbl['case_id']] #indexes out the correct patient samples

# gene expression dataset
gene = pd.read_csv('GeneExp_selected.csv') #reads in dataset
gene = gene.set_index(gene.columns[0]) #changes first column to be indices of the dataframe
gene = gene[lbl['case_id']] #indexes out the correct patient samples

# removes all non-coding transcripts from the mRNA dataset
fixed_indicies = []
for i in range(gene.shape[0]):
    fixed_indicies.append(gene.index.values[i].split('.')[0])
gene.index = fixed_indicies
coding = pd.read_csv('pr_coding_feats.csv')
keepers = list(coding['ensembl_gene_id'])
keep_ndx = []
for i in range(gene.shape[0]):
    if gene.index.values[i] in keepers:
        keep_ndx.append(i)
gene = gene.iloc[keep_ndx,:]

# CNV dataset
CNV = pd.read_csv('CNV_selected.csv') #reads in dataset
CNV = CNV.set_index('Gene_Symbol') #changes first column to be indices of the dataframe
CNV = CNV[lbl['case_id']] #indexes out the correct patient samples

# DNA methylation dataset
meth = pd.read_csv('DnaMeth_selected.csv') #reads in dataset
meth = meth.set_index('Composite Element REF') #changes first column to be indices of the dataframe
meth = meth[lbl['case_id']] #indexes out the correct patient samples

os.chdir("C:\\Users\\jonat\\Documents\\Spring 2019 Classes\\4813\\Processed_data")
gene.to_csv('gene.csv')
meth.to_csv('meth.csv')
CNV.to_csv('CNV.csv')
miRNA.to_csv('miRNA.csv')

