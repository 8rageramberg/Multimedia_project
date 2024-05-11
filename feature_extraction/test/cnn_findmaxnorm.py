import os
import sys
import numpy as np
sys.path.append("feature_extraction")
from extractors.cnn import cnn  # Import your CNN feature extractor class correctly

def extract_features_and_find_max_norm(directory):
    # Dictionary to store the path and features
    features_dict = {}

    # Initialize the feature extractor
    

    # Variable to keep track of the maximum norm and corresponding image
    max_norm = 0
    max_norm_image = None

    # Walk through all directories and subdirectories
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(subdir, file)
                extractor = cnn(full_path)
                img_features = extractor.get_features()  # Extract features
                norm = np.linalg.norm(img_features)  # Compute the norm of the feature vector
                features_dict[full_path] = img_features
                print(f"Extracted features for {full_path}, max norm is {max_norm}")
                
                # Update the maximum norm if the current one is greater
                if norm > max_norm:
                    max_norm = norm
                    max_norm_image = full_path

    return max_norm, max_norm_image

# Example usage
if __name__ == "__main__":
    # Specify the path to your dataset
    dataset_directory = r'C:\Users\fg-er\Documents\Selman\Uni\usyd\multret\Multimedia_project\archive'

    # Extract features from all images in the directory and find the maximum norm
    max_norm, max_norm_image = extract_features_and_find_max_norm(dataset_directory)
    
    # Output the results
    print(f"The image with the highest VGG16 feature norm is {max_norm_image} with a norm of {max_norm}")
