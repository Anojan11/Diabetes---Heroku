import uvicorn
from fastapi import FastAPI
from diabetes import Diabetes
import numpy as np
import pickle
import joblib
from joblib import load
import pandas as pd


app = FastAPI()

pickle_in = open("diabetes_model.sav", "rb")
classifier = pickle.load(pickle_in)


@app.get('/')
def index():
    return {'message': 'Hello, World'}



@app.get('/{name}')
def get_name(name: str):
    return{'Welcome To My Medic Website': f'{name}'}

@app.post('/predict')
def predict_diabetes(data:Diabetes):
    data = data.dict()
    Pregnancies=data['Pregnancies']
    Glucose=data['Glucose']
    BloodPressure=data['BloodPressure']
    SkinThickness=data['SkinThickness']
    Insulin =data['Insulin']
    BMI=data['BMI']
    DiabetesPedigreeFunction=data['DiabetesPedigreeFunction']
    Age=data['Age']

    sc=joblib.load('std_scaler.bin')
    scaled_test_values = sc.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

    #print(classifier.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]))
    prediction = classifier.predict(scaled_test_values)

    if(prediction[0]>0.5):
        prediction= 1
    else:
        prediction= 0
    return {
        'prediction': prediction
    }



    
    

if __name__ =='__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

#uvicorn app:app --reload
