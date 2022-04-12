from flask import Flask, render_template,request, send_from_directory, send_file
import tensorflow
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import gdown

url = 'https://drive.google.com/uc?id=1sqLURQg3HrJbgQ6DmHnI32W8-moOQLKJ'
model_path = f'{__path__[0]}/models/colorize.h5'
gdown.download(url, model_path, quiet=False)

app = Flask(__name__)
print(__path__)
model = tensorflow.keras.models.load_model(model_path)


@app.route('/api/colorization',methods=['POST'])
def colorize_image():
    image = request.files['imageUpload']
    image = np.double(
            Image.open(image).convert('L').resize((160, 160)) # image resizing,
        )
    image = np.stack((image,)*3, axis=-1)
    global pred_count
    image = np.reshape(image,(1,160,160,3))
    image = image/255.0
    output_image = model.predict(image)
    single_image = np.clip(output_image, 0.0,1.0).reshape(160,160,3)
    plt.imshow(single_image)
    plt.axis('off')
    pred_count+=1
    plt.savefig(f'{__path__[0]}/static/predictions/prediction.jpg')
    return 'success'

@app.route('/api/downloads/',methods=['GET'])
def download():
    path = f'{__path__[0]}/static/predictions/prediction.jpg'
    return send_file(path, attachment_filename=f'predcited.jpg')

@app.route('/')
def render_html():
    return render_template('index.html')


def create_app():
    return app

