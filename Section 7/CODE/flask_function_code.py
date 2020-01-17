import logging
import pathlib
import jsonify
import azure.functions as func
import os
import numpy as np
from tensorflow.keras.models import load_model

#model = load_model('price_model.h5')
model = load_model(pathlib.Path(__file__).parent / "price_model.h5")

model._make_predict_function()

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    data = req.get_json()
    value = np.array([data['crim'], 
    data['zn'],
    data['indus'], 
    data['chas'], 
    data['nox'], 
    data['rm'],
    data['age'],
    data['dis'],
    data['rad'],
    data['tax'],
    data['ptratio'],
    data['b'],
    data['lstat']])

    pred = model.predict(np.array([value]))

    prediction = {'price':int(pred[0])}

    return func.HttpResponse(body=json.dumps(prediction), status_code=200)