######## TESTING OF sift_desc_detector.py ##########
import sys
import os
sys.path.append("feature_extraction")
from extractors.sift_desc_detector import sift_desc_detector 


###### TEST 1 ######
#image_to_be_used = "archive/pull up/pull up_g20.jpg"
#new_detector = sift_desc_detector(image_to_be_used)
#print(new_detector.get_features().shape)

for root, dirs, files in os.walk("/Users/tobiashusebo/Desktop/UIB_Datasci/Sjette_semester/FAG/COMP4425/PROJECT/PROJECT_FINAL/Multimedia_project/archive"):
    for dir in dirs:
        old_name = os.path.join(root, dir)
        new_name = os.path.join(root, "_".join(dir.split(" ")))
        os.rename(old_name, new_name)


# TODO: Også sjekk et query bilde opp mot 3 andre bilder!!!

########################################################################

#In the context of feature detection algorithms like SIFT, SURF, or FAST, the term
#"response strength" refers to a measure of how distinctive or prominent a particular feature
#or keypoint is in an image.

# Access response value of the first keypoint
# first_keypoint_response = keypoints[0].response

########################################################################

# So my idea is this:

#Calculate 40 keypoints for each and every photo in a database.
# These keypoints should be the 40 keypoints with the most "response strength".
# Afterwards I want to reduce the dimensionality of it from 128 to lets say 20 or 30 
# (depending on if the cumulative variance explained by 30 is atleast 80%). 
# Then I want to send a picture as a prompt. this follows the same manners as before and is
# reduced to a 40x30 array. Then I want to calculate the distance between all of its vectors
# to other pictures with a specified threshold. 

# The picture with the most matching vectors (or descriptors) gets chosen 
# and returned to the user.

#Is this a good approach? Why / why not? 