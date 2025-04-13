import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn


scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
model = pickle.load(open("Models/model.pkl", 'rb'))


class_names = ['Database Administrator', 'Hardware Engineer', 'Application Support Engineer', 'Cyber Security Specialist',
               'Networking Engineer', 'Software Developer', 'API Specialist', 'Project Manager', 'Information Security Specialist',
               'Technical Writer', 'AI ML Specialist', 'Software tester', 'Business Analyst', 'Customer Service Executive',
               'Data Scientist', 'Helpdesk Engineer', 'Graphics Designer']


app = FastAPI()

class UserInput(BaseModel):
    personal_interests: str
    Computer_Architecture: str
    Leadership_Experience: str
    Cyber_Security: str
    Networking: str
    Software_Development: str
    Programming_Skills: str
    Project_Management: str
    Computer_Forensics_Fundamentals: str
    Technical_Communication: str
    AI_ML: str
    Software_Engineering: str
    Business_Analysis: str
    Communication_Skills: str
    Data_Science: str
    Troubleshooting_Skills: str
    Graphics_Designing: str
    


def map_experience(value):
    mapping = {"Excellent": 3, "Average": 2, "Beginner": 1, "Not Interested": 0}
    return mapping.get(value, 0)

@app.get("/")
def read_root():
    return JSONResponse(content={"message": "Welcome to the recommendation API! Please use POST /recommend to get recommendations."})

@app.post("/recommend")
def Recommendations(user_input: UserInput):
    
    feature_array = np.array([[
        map_experience(user_input.personal_interests),
        map_experience(user_input.Computer_Architecture),
        map_experience(user_input.Leadership_Experience),
        map_experience(user_input.Cyber_Security),
        map_experience(user_input.Networking),
        map_experience(user_input.Software_Development),
        map_experience(user_input.Programming_Skills),
        map_experience(user_input.Project_Management),
        map_experience(user_input.Computer_Forensics_Fundamentals),
        map_experience(user_input.Technical_Communication),
        map_experience(user_input.AI_ML),
        map_experience(user_input.Software_Engineering),
        map_experience(user_input.Business_Analysis),
        map_experience(user_input.Communication_Skills),
        map_experience(user_input.Data_Science),
        map_experience(user_input.Troubleshooting_Skills),
        map_experience(user_input.Graphics_Designing)
    ]])

    
    if feature_array.shape[1] != 17:
        return {"error": "The number of inputs does not match the model's expectations"}

    
    scaled_features = scaler.transform(feature_array)

    probabilities = model.predict_proba(scaled_features)
    top_classes_idx = np.argsort(-probabilities[0])[:3]
    recommendations = [{"job": class_names[idx], "accuracy": float(probabilities[0][idx])} for idx in top_classes_idx]

    return {"recommendations": recommendations}

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply() 
    uvicorn.run(app, host="127.0.0.1", port=8000)
