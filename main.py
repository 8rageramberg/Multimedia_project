# Imports
import os
from flask import Flask, render_template, request


##### RUN TO START APPLICATION: #####
# Local server init w/ flask:
app = Flask(__name__, static_url_path='/static')
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def save_image():
    if request.method == 'POST':
            image = request.files['image']
            if image:
                # Save the uploaded file to the static directory
                filename = 'uploaded_image.jpg'
                filepath = 'static'
                image.save(os.path.join(filepath, filename))
                return render_template('index.html', filename=filename)
            else:
                return 'No image provided in the request', render_template('index.html')

def run_app():
    app.run(port=8000, debug=True) # If needed you can change port here

if __name__ == '__main__':
    run_app()  # Click 'RUN', and go to http://127.0.0.1:8000 #change port here also if needed
