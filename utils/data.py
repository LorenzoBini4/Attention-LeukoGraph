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
for i in range(1,31):
    p[i-1]=dict()
    for ch in ['G','K','L','M','N','O','P']:
        p[i-1][ch]=FlowCal.io.FCSData(f'/folder/Case{i}_'+ch+'.fcs')

# Total features
column=('FS INT', 'SS PEAK', 'SS INT', 'SS TOF', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5', 'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB', 'FL10 INT_CD45-KO', 'TIME')

for i in range(30):   
    df_G=pd.DataFrame(p[i]['G'],columns=column)
    df_K=pd.DataFrame(p[i]['K'],columns=column)
    df_L=pd.DataFrame(p[i]['L'],columns=column)
    df_M=pd.DataFrame(p[i]['M'],columns=column)
    df_N=pd.DataFrame(p[i]['N'],columns=column)
    df_O=pd.DataFrame(p[i]['O'],columns=column)
    df_P=pd.DataFrame(p[i]['P'],columns=column)

    # Extract HSPC\myeloid HSPC\lymphoid HSPC cells
    X = pd.concat([df_M,df_L])
    X = X.drop_duplicates()
    merged = df_K.merge(X, indicator=True, how='outer')
    df_H = merged[merged['_merge'] == 'left_only']
    df_H = df_H.drop(columns=['_merge'])  

    # Add new column called 'label'    
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
    
    labels = df['label']

    # Compute weights for a weighted loss function
    class_weight = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    total_weights[i,:] = class_weight

# Save the weights in class_weights   
class_weights = total_weights.mean(axis=0)
