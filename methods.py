from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.engine.topology import Layer
from keras import backend as K
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
import glob
import copy

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import math

# %load_ext autoreload
# %autoreload 2
font = cv2.FONT_HERSHEY_SIMPLEX

## "image_to_embedding" function pass an image to the Inception network 
## to generate the embedding vector.
def image_to_embedding(image, model):
    #image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    #image = Image.open(image)
    image = np.asarray(image)
    image = cv2.resize(image, (96, 96)) 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

## calculate similarity between the captured image 
## and the images that are already been stored.
def recognize_face(face_image, input_embeddings, model):
    embedding = image_to_embedding(face_image, model)    
    minimum_distance = 200
    name = ""

    first = {'name': '', 'ed': 1000, 'pc': 0}
    second = {'name': '', 'ed': 1000, 'pc': 0}
    third = {'name': '', 'ed': 1000, 'pc': 0}
    # Loop over  names and encodings.
    for (input_name, input_embedding) in input_embeddings.items():   
      euclidean_distance = np.linalg.norm(embedding-input_embedding)
      similarity = 100 - (euclidean_distance ** math.exp(euclidean_distance))*100
      print('Euclidean distance from %s is %s' %(input_name, euclidean_distance))
      print('Percentage similarity from %s is %s' %(input_name, similarity))
      #frec_small_dick[input_name] = {'ed': euclidean_distance, 'pc':similarity}
      if euclidean_distance < minimum_distance:
          minimum_distance = euclidean_distance
          name = input_name 
      if(euclidean_distance<first['ed']):
        third = copy.deepcopy(second)
        second = copy.deepcopy(first)
        first['name'] = input_name
        first['ed'] = euclidean_distance
        first['pc'] = similarity
      elif (euclidean_distance<second['ed']):
        third = copy.deepcopy(second)
        second['name'] = input_name
        second['ed'] = euclidean_distance
        second['pc'] = similarity
      else:
        third['name'] = input_name
        third['ed'] = euclidean_distance
        third['pc'] = similarity
    

    second['ed'] = str(second['ed'])
    second['pc'] = str(second['pc'])

    third['ed'] = str(third['ed'])
    third['pc'] = str(third['pc'])
    
    if first['ed'] < 0.80:
        first['ed'] = str(first['ed'])
        first['pc'] = str(first['pc'])
        return {'bestMatch': first, 'alternateMatches': [second, third]}
    else:
        first['ed'] = str(first['ed'])
        first['pc'] = str(first['pc'])
        return {'bestMatch': {}, 'alternateMatches': [first, second, third]}