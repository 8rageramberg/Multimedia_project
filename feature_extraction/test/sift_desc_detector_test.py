####################################################
######## TESTING OF sift_desc_detector.py ##########
####################################################
# Imports and path fixing:
import sys
import os
import numpy as np
sys.path.append("feature_extraction")
from extractors.sift_desc_detector import sift_desc_detector 


def setup(utilize_test_set=False):
    # retrieving the directory to use for testing
    directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory = os.path.join(directory, "archive")

    if utilize_test_set:
        directory = os.path.join(directory, "A_test_set")
    
    # Iterating through images and making a list of image paths:
    images_to_use = []
    for root, _, files in os.walk(directory):
        for file in files:
            absolute_file_path = os.path.join(root, file)
            parent_folder = os.path.basename(os.path.dirname(absolute_file_path))

            if not utilize_test_set:
                if parent_folder == "Test_plank":
                    continue
                if parent_folder == "Test_pullup":
                    continue
            
            images_to_use.append(absolute_file_path)
    return images_to_use


def initiate_SIFT(query_image, num_descs=40, ratio=0.75):
    # Set query image and initiate SIFT:
    sift_object = sift_desc_detector(query_image, nr_descriptors=num_descs, good_match_ratio=ratio)
    sift_object.get_features()

    # Retrieve features of curr object:
    return sift_object


def initiate_test(image, descs, good_match_ratio, test_nr):
        # Initiate sift and image paths:
    query_image = image
    sift_object_query = initiate_SIFT(query_image, num_descs=descs, ratio=good_match_ratio)
    image_paths = setup(utilize_test_set=True)

    # Loop through the test images and compare:
    comparisons = []
    for img_path in image_paths:
        # Initiate curr db image sift, retrieve features, compare, store and delete.
        sift_object_db_image = initiate_SIFT(img_path, num_descs=descs, ratio=good_match_ratio)
        features = sift_object_db_image.get_features()

        comparisons.append(sift_object_query.compare(features))
        del sift_object_db_image

    # Print results
    print(f"\n##### TEST NR {test_nr}: #####\n")
    print("Params:")
    print(f"Image used: {os.path.basename(image)}\nNum of descriptors: {descs}\nRatio for good match: {good_match_ratio}")
    print(f"Image retrieved: {os.path.basename(image_paths[np.argmax(np.array(comparisons))])}\n")

    print("Images used:")
    for elem in image_paths:
        print(os.path.basename(elem))
    print("\nComparisons:")
    print(comparisons)





###### TEST ######
if __name__ == "__main__":
    # Start testing script
    images_to_use = []
    while True:
        user_input_image = input("Image to use: ")
        images_to_use.append(user_input_image)

        prompt = input("Do you want to add another image (y/n)? ")
        if (prompt.lower() == "n"):
            for i in range(1, len(images_to_use)+1):
                initiate_test(images_to_use[i-1], 40, 0.75, i)
            break