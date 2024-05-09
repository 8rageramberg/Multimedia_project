# Imports
from flask import Flask, render_template


##### RUN TO START APPLICATION: #####
# Local server init w/ flask:
app = Flask(__name__, static_url_path='/static')
@app.route('/')


def index():
    return render_template('index.html')

def run_app():
    app.run(port=8000, debug=True) # If needed you can change port here

if __name__ == '__main__':
    run_app()  # Click 'RUN', and go to http://127.0.0.1:8000 #change port here also if needed
