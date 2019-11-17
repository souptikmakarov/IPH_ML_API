import tensorflow as tf
import keras
from flask import Flask,render_template,url_for,request
from flask_ngrok import run_with_ngrok
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from keras.models import model_from_yaml

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.engine.topology import Layer
from keras import backend as K
from keras.models import load_model
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utils import LRN2D
import utils
import methods
#import variables
import glob

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

from methods import image_to_embedding

font = cv2.FONT_HERSHEY_SIMPLEX

app = Flask(__name__)
run_with_ngrok(app)
               
@app.route('/')
def home():
    #return render_template('home.html')
    return 'Server Works!'

@app.route('/find',methods=['POST'])
def find():
    image_file = request.files['fileKey']
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    resp = {}
    identity = {}
    criminalCharges = {}
    medicalInfo = {}
    socialInfo = {}
    image_file.save("./temp.jpg")
    image_file = cv2.imread("./temp.jpg", 1)
    #image = Image.open(image_file)

    height, width, channels = image_file.shape
    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h         
  
    face_image = image_file[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]  
    # load model
    model = load_model('model.h5')
    obj_text = codecs.open("input_embeddings.json", 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    input_embeddings = np.array(b_new)


    identity = recognize_face(face_image, input_embeddings, model)
    print(str(identity))
    person_name = str(identity)

    return person_name
    # return render_template('result.html',prediction = response)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run()

