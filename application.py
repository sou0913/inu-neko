from flask import Flask
from flask import render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow import keras
import numpy as np
from werkzeug.utils import secure_filename
import os

application = Flask(__name__) 

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPEG'])
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    image *= 2.0
    image -= 1.0
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

model = keras.models.load_model('inu.h5')

@application.route('/', methods=["GET"])
def show_entries():
    return render_template("mainpage.html")

@application.route('/predict', methods=["GET","POST"])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    image = request.files['image']
    if image.filename == '':
        return redirect(request.url)
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        image = load_and_preprocess_image(filepath)
        image = (tf.expand_dims(image, 0))
        preds = model.predict(image, steps=1)
        neko_prob = round(preds[0][0]*100)
        inu_prob = round(preds[0][1]*100)
        path = "/static/uploads/" + filename
        return render_template("result.html", neko_prob=neko_prob, inu_prob=inu_prob, path = path)


if __name__=='__main__':
    application.debug = True
    application.run()
