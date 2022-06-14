from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

modelPath = os.path.normpath(os.getcwd() + os.sep + os.pardir) + "/[3] Model Training"  '/model.h5'
dataPath = os.path.normpath(os.getcwd() + os.sep + os.pardir) + "\[2] Text Preprocessing" + "\_finalProcessed.csv"


model = keras.models.load_model(modelPath)
df = pd.read_csv(dataPath).drop(columns=['Unnamed: 0'])

n_most_common_words = 80000
max_len = 250

tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['finalText'].values)



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['message']
    seq = tokenizer.texts_to_sequences([text])
    text = pad_sequences(seq, maxlen=max_len)
    classes = model.predict(text)
    labels=["Art","Economy","Sports"]
    prediction = str(labels[np.argmax(classes)])
    return render_template('After.html',data =prediction )

if __name__ == '__main__':
    app.run(debug=True)