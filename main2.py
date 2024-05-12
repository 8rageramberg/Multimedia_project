##### IMPORT FIX: #####
import os
import numpy as np
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
subdirectories = next(os.walk(current_dir))[1]
for subdir in subdirectories:
    sys.path.append(os.path.join(current_dir, subdir))


##### MAIN2.py #####
from reverse_img_searcher_pickle import reverse_img_searcher_pickle as rev_search

rev = rev_search("./archive/barbell_biceps_curl/barbell_biceps_curl_100001.jpg", sift_w=0, pose_w=0.3, cnn_w=1)
print(rev.search())

