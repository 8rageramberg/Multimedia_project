##### IMPORTS: #####
import os
import shutil
from flask import Flask, jsonify, render_template, request
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
subdirectories = next(os.walk(current_dir))[1]
for subdir in subdirectories:
    sys.path.append(os.path.join(current_dir, subdir))
from reverse_img_searcher_pickle import reverse_img_searcher_pickle as rev_search


# Uncomment to disable SSL verification. Should not be needed 
# import ssl
# import requests                                          
# requests.packages.urllib3.disable_warnings()
# ssl._create_default_https_context = ssl._create_unverified_context


##### HELPER FUNCS: #####
def append_static_files(result, directory):
    '''
    Function for appending files to the static folder. 
    '''
    values = []
    paths = []
    filenames = []
    for filename in result[:, 1]:       # Loop the results from the search
        for root, _, files in os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive")): 
            if filename in files:
               filenames.append(os.path.join(root, filename))

    for comparison, name in zip(result[:,0], filenames):
        values.append(comparison)
        paths.append(name)
        static_filename = f"img{len(paths)-1}.jpg"
        save_path = os.path.join(directory, static_filename)
        
        shutil.copyfile(name, save_path)            # Copy the file into static directory with new name

    keyword, keyword_str = find_exercise(paths)     # Get keyword for the exercise
    video = find_video(keyword)                     # Get video from keyword, and youtube.txt list
    return values, paths, keyword_str, video


def find_exercise(paths):                           # Get the exercise from the path
    parts = paths[0].rsplit('/', 2)                 # split between two slash: no/what_we_want/no
    keyword = parts[-2]                             # get what we want
    keyword_str = " ".join(keyword.split("_"))      # If the word still have _ replace with space for nice view in app
    return keyword, keyword_str.upper()


def find_video(keyword):                                # Loops the links in from youtube.txt file and looks for the excercise with the keyword
    with open('./static/youtube.txt', 'r') as file:
        for row in file:
            excercise, url = row.strip().split(', ')
            if excercise == keyword:
                return url  
    return None

def _delete_img():                  
    directory = 'static'
    if os.path.exists(directory):
        files = os.listdir(directory)        # Get a list of all files in the directory
    
        for file in files:                    # Iterate over the files and delete those with names starting with 'uploaded_image'
            if file.startswith('uploaded_img') or file.startswith('img'):       # remove files that begins with uploaded_img and img
                os.remove(os.path.join(directory, file))


##### FLASK FUNCS: #####
app = Flask(__name__, static_url_path='/static')        # Flask app need static folder 
@app.route('/')
def index():
    return render_template('index.html')                # Render template in template folder


@app.route('/save_image', methods=['POST'])             # When called, save image to static folder
def save_image():
    _delete_img()                                       # Depending on filetype this may not be overwritten. Delete to make sure
    
    if 'uploaded_img' in request.files:                 # Check request for uploaded_img
        uploaded_img = request.files['uploaded_img']
        _, file_extension = os.path.splitext(uploaded_img.filename)
        uploaded_img.save('static/' + "uploaded_img" + file_extension)  # Adjust the save path with correct folder, name and filetype

        return 'Image uploaded successfully.'
    return 'No image uploaded.', 400


@app.route('/start_algo', methods=['GET'])              # Algorithm starts on /start_algo call
def find_match():
    directory = 'static'
    if os.path.exists(directory):
        files = os.listdir(directory)
        for file in files:
             if file.startswith('uploaded_img'):
                path = os.path.join(directory, file)
                rev = rev_search(path, sift_w=0, pose_w=0.6, cnn_w=1)
                result = rev.search()                   # Start algo on the user input image
    else:
        return jsonify(error="No matching image found")
    
    values, paths, keyword, video = append_static_files(result, directory)
    return jsonify(values=values, paths=paths, keyword=keyword, video=video)        # Return data in a jsonify format. This is good for javascript to handle

##### RUN TO START APPLICATION: #####
def run_app():
    app.run(port=8000, debug=True) # If needed, change port here and below

if __name__ == '__main__':
    run_app()  # RUN file and go to http://127.0.0.1:8000