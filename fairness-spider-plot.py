# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:42:27 2024

@author: kasha
"""
import pandas as pd


#%matplotlib qt

path2data = "C:/Ghazal/Agents/coding/results/gpt-4o-2024-08-06/2024-10-28-12000/"
fname = '12000-cases-2024-10-28-vignettes.xlsx'
df = pd.read_excel(path2data + fname)
df = df.loc[df['predicted label'].notna()]
#df.loc[df['CAN_BMI'] > 90] = None
#df = df[df['CAN_MOST_RECENT_HGT_CM']>125]
#df = df[df['CAN_MOST_RECENT_WGT_KG']>50]

df['Diagnosis'] = df['Liver Disease Primary Diagnosis']
df.loc[df['Secondary Diagnosis'] == 'Alcohol-related','Diagnosis'] = 'Alcohol-related'

columns = [' Gender', 'Ethnicity','ADI Quintile Category','race','Education','Liver Disease Primary Diagnosis','Secondary Diagnosis']

for col in columns:
    Data_Dict = df[col].value_counts()
    print('************ ' + col + ' ************')
    for key in Data_Dict.keys():
        FP_condtion_minority = (df[col] == key) & (df['label']== 0) & (df['predicted label']== 1)
        TP_condtion_minority = (df[col] == key) & (df['label']== 1) & (df['predicted label']== 1)
        FP_minority = len(df.loc[FP_condtion_minority ])
        TP_minority = len(df.loc[TP_condtion_minority ])
        N_minority = Data_Dict[key]
        
        print('('+ str(key) + '): '  +str(N_minority))
        print('       False Positive: ' + str(FP_minority))
        print('       True Positive: ' + str(TP_minority))
        print('       N group: ' + str(N_minority))
        
        FP_condtion_majority = (df[col] != key) & (df['label']== 0) & (df['predicted label']== 1)
        TP_condtion_majority = (df[col] != key) & (df['label']== 1) & (df['predicted label']== 1)
        
        FP_majority = len(df.loc[FP_condtion_majority ])
        TP_majority = len(df.loc[TP_condtion_majority ])
        N_majority = len(df) - Data_Dict[key]
        Deprivation_Index = ((FP_minority+TP_minority)/N_minority)/((FP_majority+TP_majority)/N_majority)
        

        print('       False Positive majority: ' + str(FP_majority))
        print('       True Positive majority: ' + str(TP_majority))
        print('       N group majority: ' + str(N_majority))

        print('Deprivation Index of ('+ str(key) + '): '  +str(Deprivation_Index))




import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
import pandas as pd
df = pd.DataFrame(dict(
    r=[      0.99,    0.98,    0.99,                    1.01,     1.00,                        0.95],
    theta=['Female','Latino','Most SES Deprived Area','Black', 'HIGH SCHOOL (9-12) or GED', 'Biliary']))
fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r= [0.75,1.25])

fig.show()