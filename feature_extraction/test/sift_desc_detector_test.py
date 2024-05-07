####################################################
######## TESTING OF sift_desc_detector.py ##########
####################################################
# Imports and path fixing:
import sys
import os
import numpy as np
sys.path.append("feature_extraction")
from extractors.sift_desc_detector import sift_desc_detector 


# retrieving the directory to use for testing
directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
directory = os.path.join(directory, "archive", "A_test_set")

# Iterating through images and making a list of image paths:
images_to_use = []
for root, _, files in os.walk(directory):
    for file in files:
        images_to_use.append(os.path.join(root, file))

print(images_to_use)