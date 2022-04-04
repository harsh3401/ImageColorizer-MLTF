from flask import Flask, render_template,request
import tensorflow
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
app = Flask(__name__)
MODEL_PATH = 'models/colorizer.h5'
model = tensorflow.keras.models.load_model('./models/colorizer.h5')

@app.route('/api/colorization',methods=['POST'])
def colorize_image():
    image = request.files['imageUpload']
    image = np.double(
            Image.open(image).convert('L').resize((160, 160)) # image resizing,
        )
    image = np.reshape(image,(1,160,160,1))

    # image = cv2.resize(image,(160,160))
    image = image/255.0
    output_image = model.predict(image)
    print(output_image[0])
    plt.imshow(output_image[0])
    plt.title("colorized")
    plt.savefig('testimage2.jpg')
    #cv2.imwrite('testimage1.jpg',output_image[0])
    # Image.fromarray(output_image[0]).save('testimage.jpg')
    return 'f'

    
    
        


@app.route('/')
def render_html():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, port=8080)
