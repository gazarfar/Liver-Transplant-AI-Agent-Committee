# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:27:28 2024

@author: kasha
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:14:59 2024

@author: kasha
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay,f1_score,precision_score,recall_score
from imblearn.metrics import specificity_score,sensitivity_score


def read_results(path2decisions,path2Tabulardata,path2vignettes):
    
    file = open(path2vignettes, "r")
    content = file.read()
    file.close()
    tmp = content.split('####################################\n')[1:]
    
    data = np.zeros((len(tmp),1))
    for i in range(0,len(tmp)):
        data[i,0] = int(tmp[i].split('Patient ID: ')[1].split('**')[0])    
    
    vignettes =  pd.DataFrame(data,columns = ['Patient ID'])
    vignettes['vignettes'] = tmp
    vignettes = vignettes.drop_duplicates(subset=['Patient ID'])
    
    
    file = open(path2decisions, "r")
    content = file.read()
    file.close()
    Decision = content.split('####################################\n')[1:]
    
    

    data = np.zeros((len(Decision),2))

    for i in range(0,len(Decision)):
        data[i,0] = int(Decision[i].split('Patient ID: ')[1].split('\n')[0])


    data[:,1] = data[:,1]+2
    for i in range(0,len(Decision)):
        data[i,0] = int(Decision[i].split('Patient ID: ')[1].split('\n')[0])
        if (Decision[i].count('Yes') == 1) and (Decision[i].count('No') == 0):
            data[i,1] = 1
        elif  (Decision[i].count('No') == 1) and (Decision[i].count('Yes') == 0):
            data[i,1] = 0
            
    for i in range(len(data)):
        if data[i,1] == 2:
            endestatment = Decision[i][len(Decision[i])-20:len(Decision[i])]
            if (endestatment.count('Yes') == 1) and (endestatment.count('No') == 0):
                data[i,1] = 1
            elif  (endestatment.count('No') == 1) and (endestatment.count('Yes') == 0):
                data[i,1] = 0
        
    for i in range(len(data)):
        if data[i,1] == 2:
            print('could not read this ones: ' + str(data[i,1]))
            
    df = pd.DataFrame(data,columns = ['Patient ID','predicted label'])
    df['Decision'] = Decision
        
    
    df = df.drop_duplicates(subset=['Patient ID'])
    
    
    df = pd.merge( df,vignettes, how='left', on='Patient ID')
    
    Wait_list_label = pd.read_excel(path2Tabulardata) #,usecols= ['Patient ID','PX_ID','TFL_COD','status.6mo','status.1yr'])
    Wait_list_label['label'] = 1 
    Wait_list_label.loc[Wait_list_label['TFL_COD'] == 'Rejected from Waitlist','label']=0
    Wait_list_label = Wait_list_label[Wait_list_label['Patient ID'].isin(df['Patient ID'])]
    columns = ['status.6mo','status.1yr']
    for col in columns: 
        Wait_list_label.loc[Wait_list_label[col] == 1, col]=2
        Wait_list_label.loc[Wait_list_label[col] == 0, col]=1
        Wait_list_label.loc[Wait_list_label[col] == 2, col]=0
    df = pd.merge(df, Wait_list_label, how='outer', on='Patient ID')
    #columns = ['Patient ID','PX_ID','TFL_COD','status.6mo','status.1yr','label','predicted label','Decision']
    return df#df[columns]

path2decisions= 'C:/Ghazal/Agents/coding/results/gpt-4o-2024-08-06/2024-10-28-12000/main/Decisions_All_2024_11_05.txt'
path2Tabulardata = "C:/Ghazal/Agents/Data/12000-cases-2024-10-28.xlsx"
path2vignettes ='C:/Ghazal/Agents/coding/results/gpt-4o-2024-08-06/2024-10-28-12000/clinical_vignettes_11489_cases_20241018 - 0 to 11488.txt'


path2falsepositives = 'C:/Ghazal/Agents/coding/results/gpt-4o-2024-08-06/2024-10-28-12000/main/falsepositives'
path2falsenegatives = 'C:/Ghazal/Agents/coding/results/gpt-4o-2024-08-06/2024-10-28-12000/main/falsenegatives'
 
df = read_results(path2decisions,path2Tabulardata,path2vignettes)

df.to_excel('C:/Ghazal/Agents/coding/results/gpt-4o-2024-08-06/2024-10-28-12000/main/' + '_All'  + '.xlsx')

columns = ['label','status.6mo','status.1yr']
title = ['waitlisting','Six months survival','1 Year survival']
n= 0
for col in columns: 
    print('******************** ' + title[n] +' ********************' )
    print('ACC : ' + str(accuracy_score(df[col].values, df['predicted label'].values)))
    print('Specificity: ' + str(specificity_score(df[col].values, df['predicted label'].values)))
    print('Sensitivity: ' + str(sensitivity_score(df[col].values, df['predicted label'].values)))
    print('Precision: ' + str(precision_score(df[col].values, df['predicted label'].values)))
    print('Recall: ' + str(recall_score(df[col].values, df['predicted label'].values)))
    print('F1 score: ' + str(f1_score(df[col].values, df['predicted label'].values)))
    
    False_positive = df.loc[(df[col] == 0) & (df['predicted label'] == 1)]
    False_positive.to_excel(path2falsepositives + '_' + col + '.xlsx')
    
    False_negative = df.loc[(df[col] == 1) & (df['predicted label'] == 0)]
    False_negative.to_excel(path2falsenegatives + '_' + col + '.xlsx')    
    
    cm = confusion_matrix(df[col].values, df['predicted label'].values)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    n = n + 1
    
    
