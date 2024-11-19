# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:40:17 2024

@author: kasha
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:41:54 2024

@author: kasha
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re
np.random.seed(2022)
from sentence_transformers import SentenceTransformer
TF_ENABLE_ONEDNN_OPTS=0
model = SentenceTransformer('all-mpnet-base-v2')

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

varis = ['What is patient sex?',
         'how old is the patient?',
         'What is patients blood type?',
         'what is the patient race',
         'what is patient ethnicity',
         'how tall is the patient',
         'what is the patient weight',
         'What is the patient education',
         'Do they use life support',
         'Do they use ventilator',
         'What is their functional status',
         'Do they have income',
         'What is socioeconomic status of a patients residential area',
         'Do they have insurance',
         'What is their BMI',
         'What is their primary liver diagnosis',
         'What is their primary diagnosis specifically',
         'What is their secondary liver diagnosis',
         'What is their secondary diagnosis specifically',
         'Do they have a history of diabetes',
         'Do they have a history of Peptic Ulcer',
         'Do they have a history of Corory Artery Disease',
         'Do they have a history of Drug Treated Hypertension',
         'Do hey have a History of Drug Treated COPD',
         'Do hey have a History of Cerebrovascular Disease',
         'Do hey have a History of Peripheral Vascular Disease',
         'Do they have a History of Pulmory Embolism',
        'Did they have Encephelopathy',
        'Did they have Variceal Bleeding',
        'Did they have Ascites',
        'Did they have Bacterial Peritonitis',
        'Did they have Portal Vein Thrombosis',
        'Did they have a TIPS Procedure',
        'Did they have a Previous Maligncy',
        'What was their initial MELD score',
        'What was their last MELD score',
        'What was their last Albumin',
        'What was their last Bilirubin',
        'What was their last INR',
        'What was their Last Serum Creatinine',
        'What was their Last Serum Sodium',
        'What was their AFP',
        'Did they had HCC with Macrovascular Invasion',
        'Did they had HCC with Extrahepatic Spread',
        'Did they had HCC with Resection Initially',
        'How many tumors did they have',
        'What was the size of their tumors',
        'Are they hospitalized',
        'Did they had previous upper abdominal surgery',
        'did they had portal hypertensive bleeding',
        'What was their MELD rate of change',
        'Did they have active extrahepatic malignancy',#
        'Did they have metastatic hcc',
        'Did they have severe cardiopulmonary disease',
        'Are they currently septic',
        'Do they use active etoh drugs',
        'Do they have acquired immunodeficiency syndrome',
        'Are they presistancy non compliance',
        'Do they have social support']


weights = np.ones((4,59))*0.1
socio_array = [7,11,12,13,57,58]
weights[0,socio_array] = 1


cardio_array = [19,21,22,24,25]
weights[1,cardio_array] = 1
 

surgn_array = [42,43,44,45,46,32,31,29,14,10,5,6]
weights[2,surgn_array] = 1
 
weights[3,:] = 1

titles = ['social worker','cardiologist','surgeon','hepatologist']
variables = model.encode(varis)



path2results = 'C:/Ghazal/Agents/coding/results/gpt-4o-2024-08-06/2024-10-28-12000/main/'
file = open(path2results + "Committee Discussion-20241028-SC-8210-to-8411.txt", "r")
content = file.read()
file.close()
Discussions = content.split('####################################\n')
Discussions = Discussions[1:]


Agent = np.zeros((4,len(Discussions),len(varis)))
scores = np.zeros((4,len(varis)))


i = 1
for i in range(0,len(Discussions)): 
    print(i)
    Task = Discussions[i].split('TaskOutput')
    Task = Task[1:]

    for j in range(0,4): 
        Text_output = re.search('raw=(.*), pydantic=None', Task[j]).group(1)

        Text_output = Text_output.replace(",", ".")
        Text_output = Text_output.replace("\\n", "")
        Text_output = Text_output.replace(" and", ".")
        Text_output = Text_output.replace(" or", ".")
        Text_output = Text_output.replace(" due to", ".")
        Text_output = Text_output.replace(" so", ".")
        Text_output = Text_output.split('.')
        Embedded_output = model.encode(Text_output)

        score = np.zeros((len(Embedded_output),len(variables)))
        for k in range(0,len(Embedded_output)): 
            score[k,:] = np.absolute(cosine(variables,Embedded_output[k,:]))#please check this later


        Agent[j,i,score.argmax(axis = 1) ] = score.max(axis = 1)
        Agent[j,i,:] = Agent[j,i,:]/max(Agent[j,i,:])
        
        
for j in range(0,4): 
    # scores[j,:] = np.sum(Agent[j,:,:],axis = 0)*weights[j,:]
    scores[j,:] = np.sum(Agent[j,:,:],axis = 0)
    scores[j,:] = scores[j,:]/max(scores[j,:])
    

    max_display = 20
    order = np.flip(np.argsort(scores[j,:]))
    features = np.array(varis)   

    fig, ax = plt.subplots(figsize =(16, 9))

    ax.barh(range(0,max_display),np.flip(scores[j,order[0:max_display]]),color = '#C8E0B4',edgecolor = '#000000')
    plt.yticks(range(max_display), reversed(features[order[0:max_display]]), fontsize=13) 
        # Remove axes splines
    
    plt.title(titles[j])       
        # Show Plot
    plt.show()


#np.load(outfile)