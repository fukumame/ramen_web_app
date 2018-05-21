# Flask などの必要なライブラリをインポートする
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import random
import string
from PIL import Image
import io
import os

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def random_name(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


def create_model(img_rows, img_cols, channel, nb_classes):
    global model

    if model is not None:
        return model

    input_tensor = Input(shape=(img_rows, img_cols, channel))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

    model.load_weights(os.path.join('data', 'finetuning.h5'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/send_image', methods=['POST'])
def analyze_image():
    img_file = request.files['img_file']
    if img_file and allowed_file(img_file.filename):
        image = img_file.read()
        image = Image.open(io.BytesIO(image))
        image = prepare_image(image, target=(150, 150))

        classes_jp = ['喜多方', '佐野']
        nb_classes = len(classes_jp)
        model = create_model(150, 150, 3, nb_classes)

        pred = model.predict(image)[0]
        top_indices = pred.argsort()[::-1]
        results = [[classes_jp[i], pred[i]] for i in top_indices]
        return render_template('index.html', results= results)


if __name__ == '__main__':
    app.run()
