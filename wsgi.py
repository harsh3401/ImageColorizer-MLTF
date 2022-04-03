from flask import Flask, render_template
import tensorflow

app = Flask(__name__)
MODEL_PATH = 'models/colorizer.h5'


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
