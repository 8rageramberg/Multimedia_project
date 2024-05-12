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


### TESTING TEST IMGS: ###
directory = os.path.dirname((os.path.abspath(__file__)))
directory = os.path.join(directory, "archive", "tester_imgs")

for root, _, files in os.walk(directory):
            for counter, file in enumerate(files):
                absolute_file_path = os.path.join(root, file)
                if ".DS_Store" in absolute_file_path:
                    continue

                rev = rev_search(absolute_file_path, sift_w=0, pose_w=0.07, cnn_w=1)
                print(rev.search())