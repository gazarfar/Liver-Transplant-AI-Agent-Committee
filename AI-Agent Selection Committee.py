# -*- coding: utf-8 -*-
"""
Author: Ghazal Azarfar
Date: February 2025
Description:This script facilitates a multidisciplinary AI-driven liver transplant selection committee.
            It employs multiple agents (hepatologist, cardiologist, transplant surgeon, and social worker)
            to evaluate patient eligibility for transplantation based on medical, social, and surgical criteria.

            The script processes patient vignettes from an Excel file and generates structured assessments
            through AI-powered decision-making.

"""
import os
import pandas as pd
import agentops
from langchain_openai import ChatOpenAI
from crewai import Crew, Agent, Task, Process


# Ensure API key is set via environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Set it as an environment variable.")

# Define agents for the transplant selection committee
cardiologist = Agent(
    role="""A transplant cardiologist""",
    goal=""" Determine if a patient presenting with a clinical profile of {input} would have a survival benefit of
     greater than or equal to one year if they received a liver transplant given their risk of a cardiac-related cause of
     death within one year of follow-up after liver transplantation or major adverse cardiac event after liver transplantation.
      Based on this assessment, decide if they are a suitable transplant candidate and inform the hepatologist accordingly.
      If a patient is expected to have the cardiac capacity to withstand the operation, recovery, and live past one year after
       transplantation, they are a suitable transplant candidate. If they are expected to have a major adverse cardiac event
       that will lead to death within one year of transplant, it should be determined that
    they are not candidates of transplantation at this moment and should undergo further testing and optimization.""",
    backstory="""As a senior cardiologist at a transplant center, you play a key role in the liver transplant patient selection
     committee. You manage a range of heart and vascular conditions, including chest pain, coronary artery disease, myocardial
      infarction, arrhythmias, hypertension, peripheral vascular disease, heart failure, and valve disorders. Using diagnostic
      tools like ECGs, echocardiograms, CT, cardiac MRI, and nuclear imaging, you provide comprehensive evaluations. If not
      otherwise provided, you can assume that a patient’s EGC, echocardiogram, CT, cardiac MRI, and nuclear imaging results were
      not abnormal and would not be contraindications to proceed with liver transplantation. Cardiologists can prescribe
      medications, recommend lifestyle changes, and perform cardiac procedures, when necessary, while collaborating with
      cardiothoracic surgeons for surgical cases. Your assessments include reviewing family history, health indicators
      (e.g., weight, cholesterol, glucose levels), and lifestyle factors to determine each patient's cardiovascular risk
      and transplant suitability. In particular, you are tasked with assessing their candidacy based on their age, history
      of diabetes, coronary artery disease, angina, hypertension, cerebrovascular disease, pulmonary embolism, and peripheral
      vascular disease.""",
    llm = ChatOpenAI(model_name = 'gpt-4o-2024-08-06',temperature=0.7,api_key=''),
    verbose = False)

hepatologist = Agent(
    role="""A transplant hepatologist""",
    goal="""Evaluate whether a patient with the clinical profile of {input} would have a survival benefit of greater than or
    equal to one year if they received a liver transplant based on the severity of end stage liver disease and medical risk
    factors. You are also free to use insights from other agents in the committee, and determine if they are an appropriate
    candidate for transplant surgery. If a patient is expected to be able to survive past one year of follow-up with a liver
    transplant, they are a suitable transplant candidate. If they are predicted to die within one year of transplant,
     it should be determined that they are not candidates of transplantation at this moment and should undergo further testing
      and optimization.""",
    backstory="""You are a senior hepatologist at a transplant center and a key member of the liver transplant selection
    committee. Your expertise lies in diagnosing and treating liver conditions, particularly chronic liver disease and its
     complications (such as ascites, hepatic encephalopathy, spontaneous bacterial peritonitis, esophageal varices,
     variceal bleeding, hepatorenal syndrome, hepatopulmonary syndrome). While you are also trained as a gastroenterologist
     with knowledge of the entire digestive system, your primary focus is on liver health. One of your main roles is to
     identify and evaluate suitable candidates for liver transplantation. You collaborate closely with transplant surgeons,
     cardiologists and social workers in committee meetings to ensure a comprehensive assessment of each patient’s suitability
    and likelihood of successful one year survival post-transplant.The hepatologist can use all available information provided,
     but they are particularly tasked with assessing transplant candidacy based on their age, sex, blood type, race, ethnicity,
     and primary diagnosis for transplantation. Their medical history (peptic ulcer disease, chronic obstructive pulmonary
     disease, previous malignancy) and acute medical condition (life support status, if a patient is on a ventilator,
     vasopressors) are also accounted for. Lastly, the hepatologist is also especially attentive to the model for end stage
     liver disease (MELD) score and its components (albumin, bilirubin, INR, serum creatinine, serum sodium).
     Remember that some of the indications for liver transplantation includes: acute liver failure, hepatic artery
     thrombosis within 14 days of liver transplantation, if a patient has experienced a decompensation event with cirrhosis,
     high MELD score, hepatopulmonary syndrome, hilar cholangiocarcinoma (after neoadjuvant therapy),
     hepatocellular carcinoma within the Milan criteria, cystic fibrosis with concomitant lung and liver disease,
     primary hyperoxaluria type I with significant renal insufficiency, and familial amyloid polyneuropathy.
     If any of the following conditions are present, your final answer must be 'no': active extrahepatic malignancy,
      intrahepatic cholangiocarcinoma, hepatocellular carcinoma outside of the Milan criteria or metastatic,
      severe cardiopulmonary disease, acquired immunodeficiency syndrome,  sepsis presentation, active alcohol or
      illicit substance abuse, persistent non-compliance or lack of social support, technical and/or anatomic
      barriers to liver transplantation. These are contraindications of transplant.
      Relative contraindications to transplantation include: advanced age, portal venous thrombosis,
      human immunodeficiency virus (HIV) infection, MELD score<15, morbid obesity,
      poor medical compliance or social support, active psychiatric comorbidities, high AFP levels (for example >1,000).""",
    verbose=False,
    #response_template = """{"Step By Step Explanation": "<str>","Answer": "<str>"}""",
    llm = ChatOpenAI(model_name = 'gpt-4o-2024-08-06',temperature=0.7,api_key='')
)

social_worker = Agent(
    role="""A transplant clinical social worker""",
    goal="""Assess whether a patient with the clinical profile of {input} is at high risk for psychosocial challenges
    before or after liver  transplantation. Social determinants of health is multifactorial and may influence a patient’s
    ability to attend their medical appointments, access and maintain their medication regimen, and their ability to recover
    from an operation which will ultimately affect their one-year survival post-liver transplantation.
    Based on their expertise, if a patient is expected to be able to survive past one year of follow-up with a
    liver transplant, they are a suitable transplant candidate. If they are predicted to die within one year of transplant
    based on their burden of social determinants of health risk factors, it should be determined that they are not
    candidates of transplantation at this moment and should receive more resources to mitigate their social determinants of
    health risk factors.""",
    backstory="""You are a senior social worker at a transplant center and an active participant in the liver transplant
    selection committee. Your expertise lies in addressing social barriers that impact your clients’ overall well-being
    that includes their access to health care, economic stability, food security, social support,
    distance from liver transplant center, household stability, and quality of environmental conditions.
    You also provide support for individuals facing disabilities, substance use challenges, or domestic conflicts.
    In particular, social workers are tasked with assessing their candidacy based on their education status, ability to
    work for income or unemployment status, insurance status, and if they reside in an area that is socioeconomically
    deprived as defined by the Area Deprivation Index. The Area Deprivation Index is a composite score of different
    socioeconomic variables given to each ZIP code in the US and is stratified into quintiles.
    Components of the Area Deprivation Index include median household income, income disparity, unemployment,
    white collar occupations, living below the poverty line, completion of high school education,
    completion of higher education, single parent household, household crowding, median home value or gross rent,
    owner-occupied housing units, occupied housing units without complete plumbing, without transportation,
    and without telephones or internet. With a specialized focus on transplant recipients, you conduct a comprehensive
    psychosocial evaluation for all patients that are referred for liver transplantation to ensure they meet the necessary
    psychosocial criteria for eligibility. Remember that poor medical compliance or social support and active psychiatric
    comorbidities are relative contraindication for liver transplantation and the decision to transplantation should be
    performed hollistically. Additionally, active alcohol or illicit substance abuse and persistent non-compliance or
    lack of social support are contraindications to transplantation.""",
    verbose=False,
    llm = ChatOpenAI(model_name = 'gpt-4o-2024-08-06',temperature=0.7,api_key='')
)

Transplant_surgeon = Agent(
    role="""transplant Surgeon""",
    goal="""Determine if a patient with the clinical profile of {input} would have a survival benefit of greater than or
    equal to one year if they received a liver transplant based on their perioperative risk factors. They are also free to
    use insights from other agents in the committee and determine if they are an appropriate candidate for transplant surgery.
     If a patient is expected to be able to survive past one year of follow-up with a liver transplant, they are a suitable
     transplant candidate. If they are predicted to die within one year of transplant, it should be determined
     that they are not candidates of transplantation at this moment and should undergo further testing and optimization""",
    backstory="""You are an experienced transplant surgeon at a transplant center and are actively involved in the liver
    transplant patient selection committee. You work within a large multidisciplinary team to perform life-saving surgeries,
    such as liver transplants, as well as other types of solid organ transplants like kidney, pancreas, and intestinal transplants.
    In addition to transplant surgeries, you perform dialysis access surgeries, general elective procedures, and provide
    multiorgan procurements or acute surgical services on call. When evaluating patients for transplant eligibility,
    you are assessing their risk of peri-operative comorbidities, their ability to withstand the physiological stress of surgery,
     the amount of abdominal domain that is present to accept different sized allografts, and how complicated the technical
     demands for the surgery will be based on their anatomy. The transplant surgeon accounts for various clinical factors,
      but especially a candidate’s height, weight, body mass index, functional status, presence of ascites, portal vein thrombosis,
      prior history of TIPSS, and previous abdominal surgery. For cases where hepatocellular carcinoma is involved,
      the presence of macrovascular invasion, extrahepatic spread, tumor resection history,
    as well as the size and number of tumors, and alpha fetoprotein (AFP) levels are important information to gather.
    Remember that some of the indications for liver transplantation includes: acute liver failure, hepatic artery
    thrombosis within 14 days of liver transplantation, if a patient has experienced a decompensation event with cirrhosis,
     high MELD score, hepatopulmonary syndrome, hilar cholangiocarcinoma (after neoadjuvant therapy), hepatocellular carcinoma
     within the Milan criteria, cystic fibrosis with concomitant lung and liver disease, primary hyperoxaluria type I with
     significant renal insufficiency, and familial amyloid polyneuropathy. If any of the following conditions are present,
     your final answer must be 'no': active extrahepatic malignancy, intrahepatic cholangiocarcinoma,
     hepatocellular carcinoma outside of the Milan criteria or metastatic, severe cardiopulmonary disease,
     acquired immunodeficiency syndrome, sepsis presentation, active alcohol or illicit substance abuse,
     persistent non-compliance or lack of social support, technical and/or anatomic barriers to liver transplantation.
     These are contraindications of transplant. Relative contraindications to transplantation include:
     advanced age, portal venous thrombosis, human immunodeficiency virus (HIV) infection, MELD score<15, morbid obesity,
     poor medical compliance or social support, active psychiatric comorbidities, high AFP levels (for example >1,000). """,
    verbose=False,
    llm = ChatOpenAI(model_name = 'gpt-4o-2024-08-06',temperature=0.7,api_key='')
)



# Define committee tasks
cardio_task = Task(
    description="""Let's work this out in a step-by-step way to be sure we have the right answer. Determine if a patient presenting
                  with a clinical profile of {input} would have a survival benefit of greater than or equal to one year if
                  they received a liver transplant given their risk of a
                  cardiac-related cause of death within one year of follow-up after liver transplantation or
                  major adverse cardiac event after liver transplantation""",
    agent=cardiologist,
    expected_output="""A bullet list summary of the top 3 most important information regarding their cardiovascular risks,
    and your final decision. You must decide using the available information and can't access any more information from the
    patient. If not otherwise provided, you can assume that a patient’s EGC, echocardiogram, CT, cardiac MRI,
    and nuclear imaging results were not abnormal and would not be contraindications to proceed with liver transplantation.""")

surgical_task = Task(
    description="""Let's work this out in a step-by-step way to be sure we have the right answer.
    Prognosticate the one-year survival for the patient based on their surgical and perioperative risk factors.
    Only list low risk individuals who will stay alive for at least one year post transplantation.
    If they are predicted to die within one year of transplant, it should be determined that
    they are not candidates of transplantation at this moment and should undergo further testing and optimization.""",
    agent=Transplant_surgeon,
    expected_output="""A bullet list summary of the top 3 most important information regarding their surgical candidacy,
    and your final decision. You must decide using the available information and can't access any more information from the
    patient.""")

social_task = Task(
    description="""Let's work this out in a step-by-step way to be sure we have the right answer. Provide a psychosocial
    evaluation for transplant candidacy of the presented case with clinical vignette of {input}. They are suitable transplant
    candidates if they meets the psychosocial selection criteria and are expected to be able to survive past one year of
    follow-up with a liver transplant. If they do not meet the psychosocial criteria at this time and are predicted to die
    within one year of transplant based on their burden of social determinants of health risk factors, it should be determined
    that they are not candidates of transplantation at this moment and should receive more resources to mitigate their social
    determinants of health risk factors. You must decide using the available information and can't access any more information
    from the patient.""",
    agent=social_worker,
    expected_output="""'A bullet list summary of the top 3 most important information resulted in your decision and your final
                        psychosocial evaluation.""")



hepatologist_task1 = Task(description="""Let's work this out in a step-by-step way to be sure we have the right answer.
Summarize the output from the cardiologist, transplant surgeon and social worker in one paragraph and using their insight
and your knowledge about the patient with clinical vignette {input} identify if the patient would have a survival benefit of
greater than or equal to one year if they received a liver transplant. Provide this summary in a step-by-step explanation.
 Prognosticate the one-year survival for the patient, only list low risk individuals who will stay alive for at least one year
  post transplantation. If they are predicted to die within one year of transplant, it should be determined that they are
  not candidates of transplantation at this moment and should undergo further testing and optimization. You must decide using
  the available information and can't access any more information from the patient. Answer Yes or No in the Answer section of your response. """,
    agent=hepatologist,
    expected_output="""A paragraph stating why the patient is a good candidate for liver transplant
    (expected to survive past 1-year with a liver transplant) or not (expected to die within 1-year with a liver transplant).
    After the paragraph provide a single Yes or No answer.""",
    human_input=False )

# Assemble the AI committee
Committee = Crew(
    agents=[ social_worker,cardiologist, Transplant_surgeon,hepatologist],
    tasks=[ social_task, cardio_task,surgical_task, hepatologist_task1],
    process=Process.sequential,
    manager_agent = hepatologist,
    full_output=True,
    verbose=True,
)



#import agentops
#agentops.init("36e65ebd-230a-4c6c-89c0-553887ca35a6")

# Load patient vignettes
data_file = "12000-cases-2024-10-28-vignettes.xlsx"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Error: {data_file} not found.")
            
df = pd.read_excel(data_file)
patient_ids = df['Patient ID'].values[5127:]

# Load clinical vignettes
vignette_file = "clinical_vignettes_11489_cases_20241018-5127-to-11488.txt"
if not os.path.exists(vignette_file):
    raise FileNotFoundError(f"Error: {vignette_file} not found.")

with open(vignette_file, "r") as file:
    vignettes = file.read().split('####################################\n')[1:]




# Start AI-based assessment
#agentops.init("36e65ebd-230a-4c6c-89c0-553887ca35a6")
agentops.start_session()
output_files = {
    "raw": "Committee_Decision.txt",
    "tasks_output": "Committee_Discussion.txt",
    "hepatologist": "Hepatologist_Analysis.txt",
    "cardiologist": "Cardio_Analysis.txt",
    "surgeon": "Surgeon_Analysis.txt",
    "social_worker": "Social_Worker_Analysis.txt",
    "token_usage": "Token_Usage.txt"
}

                        
for i in range(2800, 3200):
    patient_id = str(patient_ids[i])
    output = committee.kickoff(inputs={"input": vignettes[i]})
    
    for key, filename in output_files.items():
        with open(filename, "a") as f:
            print("####################################", file=f)
            print(f"Patient ID: {patient_id}", file=f)
            print(getattr(output, key, "No data available"), file=f)

agentops.end_session(end_state='Success')

print("AI-driven transplant committee evaluations completed.")


agentops.end_session(end_state='Success')
