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

estimator_query_img = pose_estimator(input_path)
estimator_query_img.get_features()
estimator_for_comparison = pose_estimator()

# Retrieving the directory to use for testing
directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
directory = os.path.join(directory, "archive")
directory = os.path.join(directory, "A_test_set")

for root, dirs, files in os.walk(directory):
    if ".DS_Store" in files:
        continue

    for file_name in files:
        # Create the full path to each file
        file_path = os.path.join(root, file_name)

        # Comparison:
        estimator_for_comparison.set_new_photo(file_path)
        estimator_for_comparison.get_features()
        result = estimator_query_img.compare(estimator_for_comparison.own_keypoints)
        print(f'{file_path} Is {result:.2f} % match')