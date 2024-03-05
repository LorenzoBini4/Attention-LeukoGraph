#### Hierarchical Multilabel Classification ###
'''
This script generate dataset from FCS files of 30 patients.

The notations correspond to the labels in the following way:
'O' -> T lymphocytes
'N' -> B lymphocytes
'G' -> Monocytes
'P' -> Mast cells
'K' -> HSPC 
'M' -> Myeloid HSPC 
'L' -> Lymphoid HSPC
'''
import FlowCal
import pandas as pd
import numpy as np
import os


p=dict()
for i in range(1,7):
    p[i-1]=dict()
    for ch in ['A','G','K','L','M','N','O','P']:
        p[i-1][ch]=FlowCal.io.FCSData(f'/home/users/b/bini/gnn/multiclass/folder1/Case{i}_'+ch+'.fcs')


for i in range(7,30):
    p[i-1]=dict()
    for ch in ['A','G','K','L','M','N','O','P']:
        p[i-1][ch]=FlowCal.io.FCSData(f'/home/users/b/bini/gnn/multiclass/folder2/Case{i}_'+ch+'.fcs')

p[29]=dict()
p[29]['A'] = np.genfromtxt('/home/users/b/bini/gnn/multiclass/folder2/Case31_A.csv', delimiter=";")[1:]
p[29]['G'] = np.genfromtxt('/home/users/b/bini/gnn/multiclass/folder2/Case31_G.csv', delimiter=";")[1:]
p[29]['K'] = np.genfromtxt('/home/users/b/bini/gnn/multiclass/folder2/Case31_K.csv', delimiter=";")[1:]
p[29]['L'] = np.genfromtxt('/home/users/b/bini/gnn/multiclass/folder2/Case31_L.csv', delimiter=";")[1:]
p[29]['M'] = np.genfromtxt('/home/users/b/bini/gnn/multiclass/folder2/Case31_M.csv', delimiter=";")[1:]
p[29]['N'] = np.genfromtxt('/home/users/b/bini/gnn/multiclass/folder2/Case31_N.csv', delimiter=";")[1:]
p[29]['O'] = np.genfromtxt('/home/users/b/bini/gnn/multiclass/folder2/Case31_O.csv', delimiter=";")[1:]
p[29]['P'] = np.genfromtxt('/home/users/b/bini/gnn/multiclass/folder2/Case31_P.csv', delimiter=";")[1:]

# Dataset generation Hierarchical approach

column=('FS INT', 'SS PEAK', 'SS INT', 'SS TOF', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5', 'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB', 'FL10 INT_CD45-KO', 'TIME')
for i in range(30):
    
    df_A=pd.DataFrame(p[i]['A'],columns=column)
    df_G=pd.DataFrame(p[i]['G'],columns=column)
    df_K=pd.DataFrame(p[i]['K'],columns=column)
    df_L=pd.DataFrame(p[i]['L'],columns=column)
    df_M=pd.DataFrame(p[i]['M'],columns=column)
    df_N=pd.DataFrame(p[i]['N'],columns=column)
    df_O=pd.DataFrame(p[i]['O'],columns=column)
    df_P=pd.DataFrame(p[i]['P'],columns=column)
    
    X = pd.concat([df_M,df_L])

    X = X.drop_duplicates()

    merged = df_K.merge(X, indicator=True, how='outer')

    df_H = merged[merged['_merge'] == 'left_only']
    df_H = df_H.drop(columns=['_merge'])  
    
    
    
    df_O['label']="1"
    df_N['label']="2"
    df_G['label']="3"
    df_P['label']="4"
    df_M['label']="5_1"
    df_L['label']="5_2"
    df_H['label']="5_3"
    

    directory = "Data_hierarchical"
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = pd.concat([df_O,df_N,df_G,df_P,df_M,df_L,df_H])

    df.to_csv(f"{directory}/Case_{i+1}.csv", index=False)
