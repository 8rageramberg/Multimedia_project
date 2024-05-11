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


input_path = "/Users/brageramberg/Desktop/Multimedia_project/archive/brage_test/pushup1_brage.jpeg"


best_match_my_photo = "archive/barbell_biceps_curl/barbell_biceps_curl_4300081.jpg"

#best_match_my_photo = 'archive/A_test_set/Test_plank/plank_100051_copy.jpg'

input_path = "/Users/brageramberg/Desktop/Multimedia_project/archive/brage_test/pushup1_brage.jpeg"

estimator = pose_estimator(input_path)
first = estimator.get_features()

# Retrieving the directory to use for testing
directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
directory = os.path.join(directory, "archive")
directory = os.path.join(directory, "push_up")

count = 2
my_shit_list_of_shit_numbers = []

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



count = 2
my_shit_list_of_shit_numbers = []

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