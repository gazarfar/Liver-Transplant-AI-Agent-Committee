# -*- coding: utf-8 -*-
"""
Author: Ghazal Azarfar
Date: February 2025
Description: This script processes patient data from an Excel file and generates clinical vignettes
             using an AI model. The vignettes assist liver transplant selection committees in evaluating
             patient eligibility.

"""
import concurrent.futures
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from crewai import Crew, Agent, Task
#import agentops

# Ensure API key is set via environment variable
os.environ["OPENAI_API_KEY"] = ""


# Define the Hepatologist agent
Hepatologist1 = Agent(
  role="""Hepatologist at a transplant department""",
  goal="""Write a clinical vignettes for liver transplant selection commitee meeting using {input}""",
  backstory="""You are working in transplant department, and you are an expert in writing clinical vignettes.
               you write report of the patients to be evaluated by the liver transplant selection committee.
               The committee uses your reports to decide about the patients 1 year survival and whether to include the patients
              in the transplant waitlist or not.""",
  llm = ChatOpenAI(model_name = 'gpt-4o',temperature=0.1,api_key=''),
  verbose = False)

# Define the report writing task
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


# Create the AI crew with the defined agent and task
my_crew = Crew(
    agents=[Hepatologist1],
    tasks=[report_writing],
    full_output=True,
    verbose=False,
)


# Load patient data from an Excel file
df = pd.read_excel("12000-cases-2024-10-28-vignettes.xlsx")


# Process patient IDs from a specific subset (adjustable as needed)
Patient_IDs = df['Patient ID'].values[0:1]

# Function to generate a clinical vignette for a single patient
def generate_vignette(patient):
    try:
        patient_data = []
        for col in df.loc[df['Patient ID'] == patient]:
            patient_data.append(f"{col}: {df.loc[df['Patient ID'] == patient][col].values[0]}")

        result = my_crew.kickoff(inputs={"input": patient_data})

        return f"####################################\n{result}\n"
    
    except Exception as e:
        return f"Error processing Patient ID {patient}: {str(e)}\n"

# Parallel execution using ThreadPoolExecutor
output_file = "clinical_vignettes_output.txt"

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(generate_vignette, Patient_IDs))

# Save all outputs to a file
with open(output_file, "w") as f:
    f.writelines(results)

print(f"Clinical vignettes generated and saved to {output_file}")
