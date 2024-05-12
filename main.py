#main

# Imports
import os
from flask import Flask, jsonify, render_template, request
from reverse_img_searcher import reverse_img_searcher



##### RUN TO START APPLICATION: #####
# Local server init w/ flask:
app = Flask(__name__, static_url_path='/static')
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/save_image', methods=['POST'])
def save_image():
    _delete_img()
    
    if 'uploaded_img' in request.files:
        uploaded_img = request.files['uploaded_img']
        filename, file_extension = os.path.splitext(uploaded_img.filename)
        uploaded_img.save('static/' + "uploaded_img" + file_extension)  # Adjust the save path as needed

        return 'Image uploaded successfully.'

    return 'No image uploaded.', 400

def _delete_img(): 
    directory = 'static'
    print("bajs")
    # Check if the directory exists
    if os.path.exists(directory):
        # Get a list of all files in the directory
        files = os.listdir(directory)
    
        # Iterate over the files and delete those with names starting with 'uploaded_image'
        for file in files:
            if file.startswith('uploaded_img'):
                os.remove(os.path.join(directory, file))

@app.route('/start_algo', methods=['POST'])
def find_match():
    directory = 'static'
    if os.path.exists(directory):
        files = os.listdir(directory)
        for file in files:
             if file.startswith('uploaded_img'):
                path = os.path.join(directory, file)
                rev = reverse_img_searcher(path, sift_w=1, pose_w=0, cnn_w=0)
                result = rev.search()
    else:
        return print("There is something wrong, file was not found")
    
    values = []
    paths = []

    for comparison in result:
        comparison_value, photo_name = comparison
        values.append(comparison_value)
        paths.append(photo_name)

    return jsonify(values=values, paths=paths)

  

def run_app():
    app.run(port=8000, debug=True) # If needed you can change port here



if __name__ == '__main__':
    run_app()  # Click 'RUN', and go to http://127.0.0.1:8000 #change port here also if needed
