import tensorflow as tf
import keras
import codecs, json
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
#from utils import LRN2D
import utils
import methods
#import variables
import glob
from flask import jsonify
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

from methods import *
import pickle

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
    # open a file, where you stored the pickled data
    file_self = open('important_us', 'rb')
    file_arrested = open('important_arrested', 'rb')
    file_wanted = open('important_wanted', 'rb')

    # dump information to that file
    input_embeddings_us = pickle.load(file_self)
    input_embeddings_arrested = pickle.load(file_arrested)
    input_embeddings_wanted = pickle.load(file_wanted)
    person_name = "Not a match"
    resp = {}
    identity = {}
    criminalCharges = {}
    medicalInfo = {}
    socialInfo = {}
    image_file.save("./temp.jpg")
    image_file = cv2.imread("./temp.jpg", 1)
    #image = Image.open(image_file)
    # load model
    model = load_model('model.h5', custom_objects = { "tf": tf })

    height, width, channels = image_file.shape
    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h         
  
        face_image = image_file[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]  

        dick = {}
        for idx, embedding in enumerate([input_embeddings_us,input_embeddings_wanted,input_embeddings_arrested]):
            print(len(embedding))
            identity = recognize_face(face_image, embedding, model)
            print(str(identity))
            #person_name = str(identity)
            #print(person_name)
            dick["Us" if idx == 0 else "wanted" if idx == 1 else "arrested"] = identity
            dicks= json.dumps(dick)
    return jsonify(dick)

    # identity = recognize_face(face_image, input_embeddings_wanted, model)
    # if identity is not None:
    #   print(str(identity))
    #   return str(identity)
    #   # person_name = str (embedding) + ":" + str(identity)
    #   # print(person_name) 
    # else:
    #   print("Not a match")
    #   return ("Not a match")

    # return render_template('result.html',prediction = response)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run()

