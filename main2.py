##### IMPORT FIX: #####
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
subdirectories = next(os.walk(current_dir))[1]
for subdir in subdirectories:
    sys.path.append(os.path.join(current_dir, subdir))


##### MAIN2.py #####
from reverse_img_searcher import reverse_img_searcher

rev = reverse_img_searcher("./archive/hammer_curl/hammer_curl_100001.jpg")
rev.search()