import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from matplotlib.image import imsave, imread
app = Flask(__name__)

from keras.models import load_model
from keras.backend import set_session
from skimage.transform import resize
import matplotlib.pyplot as plt
#import tensorflow as tf
import numpy as np
import cv2

print("Loading model")
global sess
sess = tf.Session()
set_session(sess)
global model
model = load_model('cnn-initial.h5')
global graph
graph = tf.get_default_graph()


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        backSub = cv2.createBackgroundSubtractorMOG2()
        file1 = request.files['file1']
        file1.save(os.path.join('uploads/'+'1.jpg'))
        f1 = cv2.imread('uploads/1.jpg')
        fgMask = backSub.apply(f1)
        file2 = request.files['file2']
        file2.save(os.path.join('uploads/'+'2.jpg'))
        f2 = cv2.imread('uploads/2.jpg')
        fgMask = backSub.apply(f2)
        file3 = request.files['file3']
        file3.save(os.path.join('uploads/' + '3.jpg'))
        f3 = cv2.imread('uploads/3.jpg')
        fgMask = backSub.apply(f3)
        cv2.imwrite(os.path.join('uploads/'+'bg.png'), fgMask)
        #resizing and merging image 1 and mask
        dim = (256,256)
        f1 = cv2.resize(f1,dim)
        fgMask =cv2.resize(fgMask,dim)
        fig_4ch = np.dstack((f1,fgMask))
        with graph.as_default():
            set_session(sess)
            probabilities = model.predict(np.array([fig_4ch,]))[0,:]
            print(probabilities)
            predictions = {"class1":number_to_class[0],"class2":number_to_class[1],"prob1":probabilities[0],"prob2":probabilities[1]}
        return render_template('predict.html', predictions=predictions)
    return render_template('index.html')

app.run(host='0.0.0.0', port=80)