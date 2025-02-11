# AI-Agent Liver Transplant Commitee 
## :pushpin: Project Overview
This repository contains an AI-driven Liver Transplant Committee Decision System that evaluates patient eligibility for liver transplantation. The system processes clinical vignettes and uses AI agents representing medical professionals—including a Hepatologist, Cardiologist, Transplant Surgeon, and Social Worker—to assess one-year survival probability post-transplant.
The project is implemented using CrewAI, LangChain, and OpenAI's GPT-4o for decision-making.

## :mag: Features
✅ Multi-Agent Decision Making: Simulates a liver transplant selection committee.
✅ AI-Powered Clinical Vignettes: Generates structured case reports.
✅ Parallel Processing: Optimized patient evaluation for efficiency.
✅ Error Handling: Ensures smooth execution with input validation.
✅ Secure API Integration: Uses environment variables for API keys.

## 🚀 Installation & Setup
### :one: Clone the Repository
```
git clone https://github.com/your-username/Liver-Transplant-AI-Committee.git
cd Liver-Transplant-AI-Committee
```
### :two: Install Dependencies
Ensure Python 3.8+ is installed, then run:
```
!pip install -q crewai
!pip install -q crewai[tools]
!pip install crewai[agentops]
```
### :three: Set Up OpenAI API Key
Create a .env file (or set as an environment variable):
```
OPENAI_API_KEY=your-openai-api-key
```
## :wrench: How It Works

1. Load Patient Data: Reads an Excel file containing patient information.
2. Generate Clinical Vignettes: AI processes medical history and generates structured reports.
3. Multi-Agent Committee Review: AI simulates a committee meeting where each agent evaluates the case:
 - :hospital: Hepatologist → Assesses liver condition & MELD scores.
 - :hearts: Cardiologist → Determines cardiac risks post-transplant.
 - :gem: Transplant Surgeon → Evaluates surgical risks & feasibility.
 - :house: Social Worker → Considers social determinants & post-op support.
4. Final Decision: A consensus is reached, and the output is stored in structured files.
