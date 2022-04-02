from flask import Flask, render_template

app = Flask(__name__)

@app.route('/api/colorization')
def colorize_image():
    try:
        return "test", 200
    except:
        return "Error", 500

@app.route('/')
def render_html():
    return render_template('./index.html')


if __name__=="__main__":
    app.run(debug=True,port=8080)