import numpy as np
from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import stopwords
import string
import pickle
def textProcessor(featureRecord):
    removePunctuation = [word for word in featureRecord if word not in string.punctuation]
    sentences = "".join(removePunctuation)
    words = sentences.split(" ")
    wordNormalize = [word.lower() for word in words]
    finalWords = [word for word in wordNormalize if word not in stopwords.words("english")]
    return finalWords

app = Flask(__name__)
model = pickle.load(open('SmsDetectorModel.mdl' , "rb"))
TfIdfObject = pickle.load(open('TFIDobject.obj','rb'))
PreprocessedText =pickle.load(open('PreprocessedText.txt','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    SmsInput = input("Enter SMS: ")
    preproccessedtext = textProcessor(SmsInput)
    bowfeatures = PreprocessedText.transform(preproccessedtext)
    tfIdffeature = TfIdfObject.transform(bowfeatures)
    predLabel = model.predict(tfIdffeature)[0]  

    return render_template('index.html', prediction_text='{} is a {} SMS.'.format(SmsInput,predLabel))


if __name__ == "__main__":
    app.run(debug=True)