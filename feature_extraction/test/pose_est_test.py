import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
import sys
sys.path.append("feature_extraction")
from extractors.pose_estimator import pose_estimator 
import cv2

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

def normalize_landmarks(landmarks):
    # Extract x, y, z, and visibility values from the landmarks
    x_values = [point['X'] for point in landmarks]
    y_values = [point['Y'] for point in landmarks]
    z_values = [point['Z'] for point in landmarks]
    visibility_values = [point['Visibility'] for point in landmarks]
    
    # Normalize the values using Min-Max scaling
    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(np.array(x_values).reshape(-1, 1)).flatten()
    y_normalized = scaler.fit_transform(np.array(y_values).reshape(-1, 1)).flatten()
    z_normalized = scaler.fit_transform(np.array(z_values).reshape(-1, 1)).flatten()
    visibility_normalized = scaler.fit_transform(np.array(visibility_values).reshape(-1, 1)).flatten()
    
    return [x_normalized, y_normalized, z_normalized, visibility_normalized]

def create_feature_vector(landmarks):
    x_normalized, y_normalized, z_normalized, visibility_normalized = normalize_landmarks(landmarks)
    feature_vector = np.concatenate((x_normalized, y_normalized, z_normalized, visibility_normalized))
    return feature_vector

def calculate_distance(feature_vector1, feature_vector2):
    return euclidean(feature_vector1, feature_vector2)


# Define the path to the archive folder
archive_path = 'archive/'

# PHOTO TO TEST: 
test_1 = 'archive/deadlift/deadlift_100031.jpg'
test_1 = 'archive/lat_pulldown/lat_pulldown_g9.jpg'
      

pose_estimator_1 = pose_estimator()
tiss, tass, keypoints_1 = pose_estimator_1.get_features(test_1)
if tiss is None:
    counter = 100000
feature_vector1 = create_feature_vector(keypoints_1)




best_number = 100
best_path = ""

counter = 0
# Loop through each folder in the archive directory
for folder in os.listdir(archive_path):

    folder_path = os.path.join(archive_path, folder)
    
    if os.path.isdir(folder_path):
        # List all paths within the folder
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if counter % 100 == 0:
                    
                    print("50 images processed")
                if counter >= 4000:
                    break
                
                photo_path = os.path.join(root, file_name)

                if "lat_pulldown_g9.jpg" in photo_path:
                    counter += 1
                    continue

                # Create an instance of pose_estimator for the current file
                tiss, tass, keypoints_2 = pose_estimator_1.get_features(photo_path)

                if tiss is not None:
                    feature_vector2 = create_feature_vector(keypoints_2)
                    distance = calculate_distance(feature_vector1, feature_vector2)
                
                    if distance < best_number:
                        best_number = distance
                        best_path = photo_path
                    
                    counter += 1
                    del tiss, tass, keypoints_2
                else:
                    counter += 1
    if counter >= 4000:
        break
    
print(counter)
print("best number", best_number)
print("best path " ,best_path)

# Load the images
image1 = cv2.imread(test_1)
image2 = cv2.imread(best_path)

# Resize images to have the same height
height = max(image1.shape[0], image2.shape[0])
width1 = int(image1.shape[1] * height / image1.shape[0])
width2 = int(image2.shape[1] * height / image2.shape[0])
image1 = cv2.resize(image1, (width1, height))
image2 = cv2.resize(image2, (width2, height))

# Create a composite image by horizontally stacking the images
composite_image = cv2.hconcat([image1, image2])

# Display the composite image
cv2.imshow("Best Image and Best Path", composite_image)
cv2.waitKey(0)
cv2.destroyAllWindows()