# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




def data_stat(data, step, xlabel):
    #data_stat plots numerical variables and provide their statistical properties
    data_bar_plot = data.copy()
    filled_percentage = len(data[data.notna()])/len(data)*100
    data_bar_plot = data_bar_plot[data_bar_plot.notna()]
    data = data[data.notna()]
    data = data.values
    data_min = np.min(data)
    data_max = np.max(data)
    data_mean = np.mean(data)
    data_median = np.median(data)
    data_std = np.std(data)
    q75, q25 = np.percentile(data, [75 ,25])
    data_var = np.var(data)
   # if np.sum(data_bar_plot.unique()%1) == 0:
      #  print('here')
      #  mylabels =  data_bar_plot.unique()
      #  y = data_bar_plot.value_counts()
      #  print('********************************************************')  
     #   print(y)
     #   if len(mylabels) > 10: 
     #       width = 6*(len(mylabels)/10)
     #   else:
     #       width = 6
     #   plt.figure(figsize=(width,6))
      #  plt.bar([str(int(value)) for value in mylabels], y[mylabels], width=0.8)
        
    #else:
        #mybin = range(int(np.floor(data_min)),int(np.floor(data_max))+2, step)
    plt.figure()
    plt.hist(data)
    plt.xlim((data_min-1, data_max+2)) 
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xticks(fontsize=20)
    plt.xticks(rotation=90)
    plt.yticks(fontsize=20)
    plt.title('min = {:.1f}'.format(data_min) + ', max = {:.1f}'.format(data_max) +', mean = {:.1f}'.format(data_mean) + ', std {:.1f}'.format(data_std)  
              + ', filled = {:.1f}'.format(filled_percentage) + ', median = {:.1f}'.format(data_median) + 'q75 = {:.1f}'.format(q75)+ 'q25 = {:.1f}'.format(q25),fontsize=20)
    plt.show()
    print('min = {:.1f}'.format(data_min) + ', max = {:.1f}'.format(data_max) +', mean = {:.1f}'.format(data_mean) + ', std {:.1f}'.format(data_std))

    #plt.savefig( xlabel +'.png')
    return 0




def data_pichart(data, mytitle):
    #data_stat plots cetegorial variables and provide their statistical properties
    total = len(data)
    filled_percentage = len(data[data.notna()])/len(data)*100
    data = data[data.notna()]
    mylabels =  data.unique()
    y = data.value_counts()
    percentage = y[mylabels]/np.sum(y[mylabels])*100
    plt.figure()
    plt.title(mytitle +', filled = {:.1f}'.format(filled_percentage) + '%' +', count = {:.0f}'.format(len(data)))
    plt.pie(y[mylabels], labels = mylabels)
    plt.show()     
    #plt.savefig(mytitle +'.png')
    labels = sorted(mylabels)
    print(labels)
    for i in range(0,len(percentage)): 
        if type(labels[i]) == str:
            print(labels[i] + ': ' + str(y[labels[i]]) + '( '+ str(y[labels[i]]/total*100)+'%)')
        else:
            print(labels[i].decode("utf-8") + ': ' + str(y[labels[i]]) + + '( '+ str(y[labels[i]]/total*100)+'%)')
       
    return 0


def visualize_dataframe(data,data_dic):
    #visualizes variables in data using the data_dic
    var_codes = list(data_dic.keys())
    var_names = list(data_dic.values())
    for j in range(0,len(data_dic)):
        tmp = data[var_codes[j]]#[var_names[j]]
        print('#' + str(j) +': var = ' + var_codes[j] + ': ' + var_names[j])
        print('data type == ' + str(tmp.dtype))
        #if 'date' in data_dic[variables[j]].lower():
        #     print('This variable is date')
        #else: 
        if tmp.dtype == 'object':  
            seq = tmp.tolist()
            seq = [value for value in seq if value != '']
            tmp = pd.DataFrame(seq, columns = [var_names[j]])
            print('count = {:.0f}'.format(len(tmp)))
            print('********************************************************')  
            print(tmp[var_names[j]][0:3])
            print('********************************************************')  
            data_pichart(tmp[var_names[j]], var_names[j])
        elif (tmp.dtype == 'float64') or (tmp.dtype == 'int64') :
            seq = tmp.tolist()
            print('count = {:.0f}'.format(len(tmp[tmp.notna()]))+', filled = {:.2f}'.format(len(tmp[tmp.notna()])/len(tmp)*100) + '%')
            print('********************************************************')  
            print(tmp[tmp.notna()][0:3])
            print('********************************************************')  
            data_stat(tmp, 1, var_names[j])
        print('#################################################################################################')
    return 0

def visualize_dataframe2(data,data_dic):
    #visualizes variables in data using the data_dic
    var_codes = list(data_dic.keys())
    var_names = list(data_dic.values())
    for j in range(0,len(data_dic)):
        tmp = data[var_codes[j]]
        print('#' + str(j) +': var = ' + var_codes[j] + ': ' + var_names[j])
        print('data type == ' + str(tmp.dtype))
        #if 'date' in data_dic[variables[j]].lower():
        #     print('This variable is date')
        #else: 
        if tmp.dtype == 'object':  
            seq = tmp.tolist()
            seq = [value for value in seq if value != '']
            tmp = pd.DataFrame(seq, columns = [var_names[j]])
            print('count = {:.0f}'.format(len(tmp)))
            print('********************************************************')  
            print(tmp[var_names[j]][0:3])
            print('********************************************************')  
            data_pichart(tmp[var_names[j]], var_names[j])
        elif (tmp.dtype == 'float64') or (tmp.dtype == 'int64') :
            seq = tmp.tolist()
            print('count = {:.0f}'.format(len(tmp[tmp.notna()]))+', filled = {:.2f}'.format(len(tmp[tmp.notna()])/len(tmp)*100) + '%')
            print('********************************************************')  
            print(tmp[tmp.notna()][0:3])
            print('********************************************************')  
            data_stat(tmp, 1, var_names[j])
        print('#################################################################################################')
    return 0
