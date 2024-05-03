######## TESTING OF sift_desc_detector.py ##########
import sys
sys.path.append("feature_extraction")
from extractors.sift_desc_detector import sift_desc_detector 


###### TEST 1 ######
image_to_be_used = "archive/pull up/pull up_g20.jpg"
new_detector = sift_desc_detector(image_to_be_used)
print(new_detector.get_features().shape)