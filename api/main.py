from fastapi import FastAPI
import dill
from pydantic import BaseModel
import pandas as pd


class Form(BaseModel):
    description: str
    fuel: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    odometer: int
    posting_date: str
    price: int
    region: str
    region_url: str
    state: str
    title_status: str
    transmission: str
    url: str
    year: int


class Prediction(BaseModel):
    id: int
    pred: str
    price: int


app = FastAPI()
with open('cars_pipe.pkl', 'rb') as file:
    loaded_data = dill.load(file)


@app.get('/status')
def status():
    return 'I`m OK'


@app.get('/version')
def status():
    return loaded_data['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    result = loaded_data['model'].predict(df)
    return {
        'id': form.id,
        'pred': result[0],
        'price': form.price
    }
