import os
import sys
import numpy as np
sys.path.append("feature_extraction")
from extractors.cnn import extractor_2  # Make sure to import your class correctly

def extract_features(directory):
    # Dictionary to store the path and features
    features_dict = {}

    # Initialize the feature extractor
    extractor = extractor_2()

    # Walk through all directories and subdirectories
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(subdir, file)
                extractor.photo_path = full_path  # Update the photo_path for the extractor
                img_features = extractor.get_features()  # Extract features
                features_dict[full_path] = img_features
                print(f"Extracted features for {full_path}")

    return features_dict

def find_closest_image(query_features, features_dict):
    # Initialize variables to track the closest image
    min_distance = float('inf')
    closest_image = None

    # Compare the query features with each image features in the dictionary
    for path, features in features_dict.items():
        distance = np.linalg.norm(query_features - features)
        if distance < min_distance:
            min_distance = distance
            closest_image = path

    return closest_image, min_distance

# Example usage
if __name__ == "__main__":
    # Specify the path to your dataset and query image
    dataset_directory = r'C:\Users\fg-er\Documents\Selman\Uni\usyd\multret\Multimedia_project\archive\A_test_set'
    query_image_path = r'C:\Users\fg-er\Downloads\neutral-grip-pull-up.jpg'  # Adjust this to your query image path

    # Extract features from all images in the directory
    dataset_features = extract_features(dataset_directory)

    # Initialize the extractor and extract features from the query image
    query_extractor = extractor_2(query_image_path)
    query_features = query_extractor.get_features()

    # Find and print the closest image
    closest_image, distance = find_closest_image(query_features, dataset_features)
    print(f"The closest image to the query is {closest_image} with a distance of {distance}")
import os
import sys
import numpy as np
sys.path.append("feature_extraction")
from extractors.cnn import extractor_2  # Make sure to import your class correctly

def extract_features(directory):
    # Dictionary to store the path and features
    features_dict = {}

    # Initialize the feature extractor
    extractor = extractor_2()

    # Walk through all directories and subdirectories
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(subdir, file)
                extractor.photo_path = full_path  # Update the photo_path for the extractor
                img_features = extractor.get_features()  # Extract features
                features_dict[full_path] = img_features
                print(f"Extracted features for {full_path}")

    return features_dict

def find_closest_image(query_features, features_dict):
    # Initialize variables to track the closest image
    min_distance = float('inf')
    closest_image = None

    # Compare the query features with each image features in the dictionary
    for path, features in features_dict.items():
        distance = np.linalg.norm(query_features - features)
        if distance < min_distance:
            min_distance = distance
            closest_image = path

    return closest_image, min_distance

# Example usage
if __name__ == "__main__":
    # Specify the path to your dataset and query image
    dataset_directory = r'C:\Users\fg-er\Documents\Selman\Uni\usyd\multret\Multimedia_project\archive\A_test_set'
    query_image_path = r'C:\Users\fg-er\Downloads\neutral-grip-pull-up.jpg'  # Adjust this to your query image path

    # Extract features from all images in the directory
    dataset_features = extract_features(dataset_directory)

    # Initialize the extractor and extract features from the query image
    query_extractor = extractor_2(query_image_path)
    query_features = query_extractor.get_features()

    # Find and print the closest image
    closest_image, distance = find_closest_image(query_features, dataset_features)
    print(f"The closest image to the query is {closest_image} with a distance of {distance}")
