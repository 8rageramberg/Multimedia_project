# Imports
import os
import shutil
from flask import Flask, jsonify, render_template, request
import os
import numpy as np
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
subdirectories = next(os.walk(current_dir))[1]
for subdir in subdirectories:
    sys.path.append(os.path.join(current_dir, subdir))
from reverse_img_searcher_pickle import reverse_img_searcher_pickle as rev_search



##### HELPER FUNCS: #####
def append_static_files(result, directory):
    '''
    Function for appending files to the static folder. 
    '''
    values = []
    paths = []
    filenames = []
    for filename in result[:, 1]:
        for root, _, files in os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive")):
            if filename in files:
               filenames.append(os.path.join(root, filename))

    for comparison, name in zip(result[:,0], filenames):
        values.append(comparison)
        paths.append(name)
        static_filename = f"img{len(paths)-1}.jpg"
        save_path = os.path.join(directory, static_filename)
        
        shutil.copyfile(name, save_path)

    keyword = find_exercise(paths)
    video = find_video(keyword)
    return values, paths, keyword, video

def find_exercise(paths):
    parts = paths[0].rsplit('/', 2)
    keyword = parts[-2]
    return keyword

def find_video(keyword):
    with open('./static/youtube.txt', 'r') as file:
        for row in file:
            excercise, url = row.strip().split(', ')
            if excercise == keyword:
                return url  
    return None  


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
    # Check if the directory exists
    if os.path.exists(directory):
        # Get a list of all files in the directory
        files = os.listdir(directory)
    
        # Iterate over the files and delete those with names starting with 'uploaded_image'
        for file in files:
            if file.startswith('uploaded_img') or file.startswith('img'):
                os.remove(os.path.join(directory, file))

@app.route('/start_algo', methods=['GET'])
def find_match():
    directory = 'static'
    if os.path.exists(directory):
        files = os.listdir(directory)
        for file in files:
             if file.startswith('uploaded_img'):
                path = os.path.join(directory, file)
                rev = rev_search(path, sift_w=0, pose_w=1, cnn_w=0)
                result = rev.search()
    else:
        return jsonify(error="No matching image found")
    
    values, paths, keyword, video = append_static_files(result, directory)
    return jsonify(values=values, paths=paths, keyword=keyword, video=video)

def run_app():
    app.run(port=8000, debug=True) # If needed you can change port here

if __name__ == '__main__':
    run_app()  # Click 'RUN', and go to http://127.0.0.1:8000 #change port here also if needed