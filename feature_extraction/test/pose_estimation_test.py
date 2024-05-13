####################################################
########## TESTING OF pose_estimator.py ############
####################################################
# Imports and path fixing:
import os
import sys
import warnings
sys.path.append("feature_extraction")
from extractors.pose_estimator import pose_estimator 
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")


test_1 = 'archive/deadlift/deadlift_100031.jpg'
test_2 = 'archive/lat_pulldown/lat_pulldown_g9.jpg'
input_path = 'archive/A_test_set/Test_plank/plank_100051_copy.jpg'
test_1 = 'archive/deadlift/deadlift_100031.jpg'
test_2 = 'archive/lat_pulldown/lat_pulldown_g9.jpg'
best_match_my_photo = "archive/barbell_biceps_curl/barbell_biceps_curl_4300081.jpg"


################################################################################################
################################################################################################
################################################################################################
# This class have been altered multiple times to fit different phases in the project. This is the last iteration of testing. This is also independent
# on other classes, meaning it is looping spesific folders and gets the feature thorugh the pose estimator class directly. This was good for debugging 
# and working independently

input_path = "/Users/brageramberg/Desktop/Multimedia_project/archive/A_test_set/Test_plank/brage_pushup.jpeg"
estimator = pose_estimator(input_path)
first = estimator.get_features(display=True)

# Retrieving the directory to use for testing
directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
directory = os.path.join(directory, "archive")
directory = os.path.join(directory, "push_up")

best_match = 0
best_path = ""

for root, dirs, files in os.walk(directory):
    if ".DS_Store" in files:
        continue

    for file_name in files:
        # Create the full path to each file
        file_path = os.path.join(root, file_name)

        # Comparison:
        estimator.set_new_photo(file_path)
        result_10 = estimator.get_features()
        result = estimator.compare(first)
        
        if result > best_match:
            best_match = result
            best_path = file_path
        
print("push up best number and path : ", best_match, "  ", best_path)

# Retrieving the directory to use for testing
directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
directory = os.path.join(directory, "archive")
directory = os.path.join(directory, "barbell_biceps_curl")

best_match = 0
best_path = ""

for root, dirs, files in os.walk(directory):
    if ".DS_Store" in files:
        continue

    for file_name in files:
        # Create the full path to each file

        
        file_path = os.path.join(root, file_name)

        # Comparison:
        estimator.set_new_photo(file_path)
        result_10 = estimator.get_features()
        result = estimator.compare(first)
        
        if result > 0:
            best_match = result
            best_path = file_path
        
print("push up best number and path : ", best_match, "  ", best_path)