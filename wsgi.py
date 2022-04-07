from flask import Flask, render_template,request, send_from_directory, send_file
import tensorflow
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

app = Flask(__name__)
model = tensorflow.keras.models.load_model('./models/colorize_3.h5')
pred_count=0


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
    plt.savefig(f'static/predictions/prediction{pred_count}.jpg')
    return 'success'

@app.route('/api/downloads/',methods=['GET'])
def download():
    path = f'static/predictions/prediction{pred_count}.jpg'
    return send_file(path, attachment_filename=f'predcited.jpg')

@app.route('/')
def render_html():
    return render_template('index.html')


if __name__ == "__main__":
    app.run()
