from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS, cross_origin
import sys
import io
from keras.models import load_model
import nltk
import numpy as np
import pickle
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#setup the flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = load_model('chatbot_model.h5')
data = pickle.load( open( "chatbot-data.pkl", "rb" ) )
words = data['words']
classes = data['classes']

model._make_predict_function()


#prediction route
@app.route('/chat_with_me', methods=["POST"])
@cross_origin()
def get_answer():
    data = request.json
    question = data['question']
    print(question)
    p = parse_sentence(question, words) 
    input = np.array([p])
    prediction = model.predict(inputvar)
    answer= classes[np.argmax(prediction)]
    prediction = {'answer':answer}

    return jsonify(prediction)

#this function is for cleanup and tokenizing
def tokenize_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def parse_sentence(sentence, words):
    sentence_results = tokenize_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_results:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return(np.array(bag))

if __name__ == "__main__":
        app.run()