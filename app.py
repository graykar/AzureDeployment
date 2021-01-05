# -*- coding: utf-8 -*-
"""
Created on Thu Dec-20-2020

@author: Gururaj Raykar
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'C://Users//graykar//Downloads//Pyhton//Python Practise//Jupyter Notebook//Deep_Learning//Deployment//CNN_Image_Classification//classify.h5'

model = load_model(MODEL_PATH)
model.summary()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(180, 180))
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    preds=np.argmax(pred, axis=1)
    if preds == 0:  
        preds = 'This is Raghu'
    if preds == 1:
        preds = 'This is Shreyas'
    else :
        preds = 'This is Suresh'
    return preds
	
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None	


if __name__ == '__main__':
    app.run(debug=True)
