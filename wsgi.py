from flask import Flask, render_template

import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
MODEL_PATH = 'models/colorizer.h5'

# Loading the trained model
model = load_model(MODEL_PATH)
model.make_predict_function()   

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/api/colorization')
def colorize_image():
    try:
        model = tensorflow.load_model('./models/colorizer.h5')
        return "test", 200
    except:
        return "Error", 500


@app.route('/')
def render_html():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, port=8080)
