from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model

app = Flask(__name__)

model = load_model('price_model.h5')

model._make_predict_function()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
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

    return jsonify(prediction)
    
if __name__ == "__main__":
    app.run(debug=True)