##### IMPORT FIX: #####
import os
import numpy as np
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
subdirectories = next(os.walk(current_dir))[1]
for subdir in subdirectories:
    sys.path.append(os.path.join(current_dir, subdir))


##### MAIN2.py #####
from reverse_img_searcher import reverse_img_searcher

# Native grid search
weights = np.arange(start=0.1, step=0.1, stop=1)

for x in weights:
    for y in weights[1:len(weights)]:
        rev = reverse_img_searcher("./archive/brage_test/pushup2_brage.jpeg", sift_w=y, pose_w=x)
        list_of_results_1 = rev.search()
        for acc, name in list_of_results_1:
            print(f"Pic: {name}, {float(acc):.2f}")
        print("################################\n\n")
