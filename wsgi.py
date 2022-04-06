from flask import Flask, render_template,request, send_from_directory, send_file
import tensorflow
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
app = Flask(__name__)
model = tensorflow.keras.models.load_model('./models/colorize_3.h5')

@app.route('/api/colorization',methods=['POST'])
def colorize_image():
    image = request.files['imageUpload']
    print(np.double(Image.open(image)).shape)
    image = np.double(
            Image.open(image).convert('L').resize((160, 160)) # image resizing,
        )
    image = np.stack((image,)*3, axis=-1)
    print(image.shape)
    image = np.reshape(image,(1,160,160,3))
    
    image = image/255.0
    output_image = model.predict(image)
    single_image = np.clip(output_image, 0.0,1.0).reshape(160,160,3)
    print(single_image.shape)
    plt.imshow(single_image)
    plt.axis('off')
    plt.savefig('testimage3.jpg')
    return 'f'

@app.route('/api/downloads/',methods=['GET'])
def download():
    path = 'testimage.jpg'
    return send_file(path, attachment_filename='testimage.jpg')

@app.route('/')
def render_html():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, port=8080)
