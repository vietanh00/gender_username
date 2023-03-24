import pickle
import numpy as np
import pandas as pd

import uvicorn
from fastapi import FastAPI

# Creating FastAPI instance
app = FastAPI()

# Load model classification and vectorizer
with open('models/rf_clf.pickle', 'rb') as m:
    rf_clf = pickle.load(m)

with open('models/vectorizer.pickle', 'rb') as vc:
    vectorizer = pickle.load(vc)

# examples = pd.read_csv('data/gender.csv')
# name = examples.sample(1)['name'].values[0]
# print(type(name))

@app.get('/predict')
async def predict_gender(name: str = 'Nguyen'):
    # name = examples.sample(1)['name'].values[0]
    name_vec = vectorizer.transform([name.lower()])
    prob = rf_clf.predict_proba(name_vec)[:, 1][0]
    gender = 'male' if prob > 0.5 else 'female'
    # print({'name': name, 'gender': gender, 'score': np.round(prob, 4)})
    return {'name': name, 'gender': gender, 'score': np.round(prob, 4)}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)