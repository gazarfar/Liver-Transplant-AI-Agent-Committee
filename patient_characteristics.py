# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:32:01 2024

@author: kasha
"""

import pandas as pd

from Utility import visualize_dataframe

#%matplotlib qt


path2data = "C:/Ghazal/Agents/coding/results/gpt-4o-2024-08-06/2024-10-28-12000/"
fname = '12000-cases-2024-10-28-vignettes.xlsx'
df = pd.read_excel(path2data + fname)
df = df.loc[df['predicted label'].notna()]

positive = df.loc[df['label'] == 1]
negative = df.loc[df['label'] == 0]

var_dictionary = {}

for i in range(40,50):
    print(str(i)+ ') ' + positive.columns[i] + ': ' + positive.columns[i] )
    var_dictionary[positive.columns[i]] = positive.columns[i]

visualize_dataframe(positive,var_dictionary)

# n = 0
# for col in df.columns[60:64]:
#    print(str(n) + ') ' + str(col[1]) + ' missigness, n (%) of {:.1f}'.format(len(df)) +', {:.1f}'.format(df[col].isna().sum())+'( {:.1f}'.format(df[col].isna().sum()/len(df)*100) + ' %)')
#    print('')
#    n +=1
    
# df = df.drop(columns = ['Artificial Liver','IV Pressors','albumin rate of change','bilirubin rate of change','INR rate of change',
#                         'Zipcode','zip AID','creatinine rate of change','serum sodium rate of change',
#                           'MELD or PELD'])
# df = df.loc[df['Last Serum Creatinine'] < 15]
# df = df.loc[df['Last INR'] < 10]
# df = df.loc[df['Last Bilirubin'] < 30]

#df_life_support = df.loc[df['CAN_LIFE_SUPPORT','Candidate Life Support'] == 'Y']




#df = df.groupby(['1 year mortality']).sample(n=200, random_state=1)
#df = df.sample(n=200, random_state=222)#accepted
#df = df.sample(n=50, random_state=444)#rejected
#df = df.sample(n=200, random_state=32)#ald
#df = df.sample(n=200, random_state=2022)#deprivedarea
#var_dictionary = {}
#for i in range(len(df.columns)-20,len(df.columns)):
#    print(str(i)+ ') ' +df.columns[i][0] + ': ' +df.columns[i][1] )
#    var_dictionary[df.columns[i][0]] = df.columns[i][1]

#visualize_dataframe(df,var_dictionary)

#df = pd.concat([df,df_life_support])

# fname = '200-cases-2024-10-18-minority-20241018.xlsx'

# df.to_excel(path2data + fname)