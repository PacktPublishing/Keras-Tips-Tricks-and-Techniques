from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS, cross_origin
import sys
import io
from keras.models import load_model

#setup the flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#load the trained model
model = load_model('shakespeare_generator.h5')

model._make_predict_function()

#optional, load the shakespeare text to get the unique character count for use in predictions.
path = 'shakespeare.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print('total chars:', len(chars))

#prediction route
@app.route('/generate_text', methods=["POST"])
@cross_origin()
def generate_text():
    data = request.json
    value = data['seed']
    print(value)
    gentext = text_generator(value)
    
    prediction = {'text':gentext}

    return jsonify(prediction)

#this function takes a seed sentence and generates 400 additional characters in the style of shakespeare
def text_generator(sentence):
    maxlen = 40
    generated = ''
    generated += sentence    
    
    for i in range(400):
        x_pred = np.zeros((1, maxlen, 79))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sampler(preds, 0.2)
        next_char = indices_char[next_index]

        sentence = sentence[1:] + next_char
        generated += next_char
    return generated

def sampler(prediction, temperature):
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temperature
    exp_prediction = np.exp(prediction)
    final_pred = exp_prediction / np.sum(exp_prediction)
    prob = np.random.multinomial(1, final_pred, 1)
    return np.argmax(prob)

if __name__ == "__main__":
        app.run()