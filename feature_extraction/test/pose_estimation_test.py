import os
import sys

sys.path.append("feature_extraction")
from extractors.pose_estimator import pose_estimator 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
# filter out warnings: /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/google/protobuf/symbol_database.py:55: UserWarning: SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead. SymbolDatabase.GetPrototype() will be removed soon.

test_1 = 'archive/deadlift/deadlift_100031.jpg'
test_2 = 'archive/lat_pulldown/lat_pulldown_g9.jpg'

input_path = 'archive/A_test_set/Test_plank/plank_100051_copy.jpg'
estimator = pose_estimator(input_path)
keypoints_1 = estimator.get_features()

# retrieving the directory to use for testing
directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
directory = os.path.join(directory, "archive")
directory = os.path.join(directory, "A_test_set")

for root, dirs, files in os.walk(directory):
    if ".DS_Store" in files:
        continue
        
    # Print the current root directory
    # print("Current directory:", root)
    # print("Subdirectories:", dirs)
    # print("Files:", files)

    for file_name in files:
        # Create the full path to each file
        file_path = os.path.join(root, file_name)
        # Print the full path of each file

        estimator.set_new_photo(file_path)
        keypoints_2 = estimator.get_features()
        result = estimator.compare(keypoints_1, keypoints_2)
        print(f'{file_path} Is {result:.2f} % match')