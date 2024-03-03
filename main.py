from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Load data and train the model
df = pd.read_csv('/content/drive/MyDrive/d drive wala/dataset/processed-data.csv')
df = df.drop(df.columns[[4, 5]], axis=1)
df.rename(columns={'Severity_None': 'Target'}, inplace=True)
x = df.drop(columns=['Target'])
y = df['Target']
model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3,
                                   min_samples_split=2, min_samples_leaf=1, random_state=42)
model.fit(x, y)

# Define FastAPI app
app = FastAPI()

# Pydantic model for request body
class SymptomsInput(BaseModel):
    tiredness: int
    dry_cough: int
    difficulty_breathing: int
    sore_throat: int
    nasal_congestion: int
    runny_nose: int
    age: int
    gender: str

# Pydantic model for response body
class PredictionResponse(BaseModel):
    severity_prediction: int
    recommendation: str

# Service class encapsulating business logic
class AsthmaService:
    @staticmethod
    def check_threshold(symptoms: List[int], age: int) -> bool:
        if all(symptoms) and age > 68:
            return True
        elif age < 11 and symptoms[2] == 1 and symptoms[0] == 1:
            return True
        else:
            return False

    @staticmethod
    def get_recommendation(severity: int) -> str:
        if severity == 0:
            return "Your asthma condition is currently under control. Continue to monitor your symptoms regularly."
        elif severity == 1 or severity == 2:
            return "You are experiencing mild to moderate asthma symptoms. Try some home remedies such as steam inhalation and staying hydrated."
        elif severity == 3:
            return "You are experiencing severe asthma symptoms. Please seek immediate medical attention. Use this link to find a chest physician or specialist in your area: [Find a Physician Link]."

# FastAPI endpoint
@app.post("/predict")
async def predict_symptoms(symptoms: SymptomsInput) -> PredictionResponse:
    # Map gender to lowercase
    gender = symptoms.gender.lower()

    # Check if the threshold condition is met
    if AsthmaService.check_threshold([symptoms.tiredness, symptoms.dry_cough, symptoms.difficulty_breathing,
                                      symptoms.sore_throat, symptoms.nasal_congestion, symptoms.runny_nose],
                                     symptoms.age):
        severity_prediction = 3
        recommendation = AsthmaService.get_recommendation(severity_prediction)
    else:
        none_experiencing = 1 if sum([symptoms.tiredness, symptoms.dry_cough, symptoms.difficulty_breathing,
                                      symptoms.sore_throat, symptoms.nasal_congestion, symptoms.runny_nose]) == 0 else 0
        user_input = [symptoms.tiredness, symptoms.dry_cough, symptoms.difficulty_breathing, symptoms.sore_throat,
                      symptoms.nasal_congestion, symptoms.runny_nose, none_experiencing,
                      1 if symptoms.age <= 9 else 0, 1 if 10 <= symptoms.age <= 19 else 0,
                      1 if 20 <= symptoms.age <= 24 else 0, 1 if 25 <= symptoms.age <= 59 else 0,
                      1 if symptoms.age >= 60 else 0, 1 if gender == 'female' else 0, 1 if gender == 'male' else 0,
                      0, 0]
        severity_prediction = model.predict([user_input])[0]
        recommendation = AsthmaService.get_recommendation(severity_prediction)

    return PredictionResponse(severity_prediction=severity_prediction, recommendation=recommendation)
