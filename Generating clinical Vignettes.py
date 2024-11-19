# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:43:55 2024

@author: kasha
"""
#   Required libraries
#pip install -q crewai
#pip install -q crewai[tools]
#pip install crewai[agentops]


import os
import pandas as pd
from langchain_openai import ChatOpenAI
from crewai import Crew, Agent, Task
#import agentops
#add the keys here



Hepatologist1 = Agent(
  role="""Hepatologist at a transplant department""",
  goal="""Write a clinical vignettes for liver transplant selection commitee meeting using {input}""",
  backstory="""You are working in transplant department, and you are an expert in writing clinical vignettes.
               you write report of the patients to be evaluated by the liver transplant selection committee.
               The committee uses your reports to decide about the patients 1 year survival and whether to include the patients
              in the transplant waitlist or not.""",
  llm = ChatOpenAI(model_name = 'gpt-4o',temperature=0.1,api_key=''),
  verbose = False)


report_writing = Task(
    description="""Using the patient data in {input}  write a report for the Liver transplant committee meeting.
    Your report include all the important information regarding the patient's health such as: patient's characteristic, medical history, and social background.
    patient ID, age, gender, patient race, height, weight, BMI, education, medical condition, blood type,
    the vignettes should include patient' extrahepative malignancy status, whether they have metastatic HCC,if they have severe cardiopulmonary, are currently septic,
    use active eroh drugs, have acquired immunodeficiency syndrome,
    liver chacteristics such as initial, and lastest MELD score, as well as the rate of change in the MELD score and
     Labs such as Albumin, bilirubin, INR, creatinine, serum sodium, AFP, severe cardiopulmonaryd,sepsis status, etoh drugs, aid,
    primary liver diagnosis, Past medial history and co-morbidities (diabetes, dialysis,peptic ulcer,hypertension, COPD, Cerebrovascular Disease, Pulmonary Embolism) ,current medications,
     include information such as insurance, ADI Quintile Category', income, social support and persistance non compliance for the social worker.
     	Ascites, Encephalopathy, Varices,
       Your report are evaluated by a committee including a hepatologist, a transplant surgen, a cardiologist and a social worker. They will use
       your report to make a decision whether to include the patient in the transplant waitlist or not.
       """,
    agent=Hepatologist1,
    expected_output="""A Clinical vignette in a paragraph format, explaining the case, including all the iformation required by the
                       transplant committee to make decision regarding the patient's listing in a liver transplant waiting
                       list. The paragrah title is the patient ID and it includes patient's characteristics,
                       clinical history and social history""")


my_crew = Crew(
    agents=[Hepatologist1],
    tasks=[report_writing],
    full_output=True,
    verbose=False,
)


df = pd.read_excel("12000-cases-2024-10-28-vignettes.xlsx")
Patient_ID = df['Patient ID'].values
Patient_ID = Patient_ID[6266:]

for patient in Patient_ID:
  Tabel = []
  for col in df.loc[df['Patient ID'] == patient]:
    Tabel.append(col + ':' + str(df.loc[df['Patient ID'] == patient][col].values))

  result = my_crew.kickoff(inputs={"input": Tabel})

  with open("clinical_vignettes_11489_cases_20241018.txt", "a") as f:
    print('####################################',file=f)
    print(result, file=f)


