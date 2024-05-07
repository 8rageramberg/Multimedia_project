import os
import cv2
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean

sys.path.append("feature_extraction")
from extractors.pose_estimator import pose_estimator 


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
# filter out warnings: /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/google/protobuf/symbol_database.py:55: UserWarning: SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead. SymbolDatabase.GetPrototype() will be removed soon.

def normalize_landmarks(landmarks):
    # Extract x, y, z, and visibility coordinates Pose estimator landmark object
    x = [data['X'] for data in landmarks]
    y = [data['Y'] for data in landmarks]
    z = [data['Z'] for data in landmarks]
    vis = [data['Visibility'] for data in landmarks]

    scaler = MinMaxScaler()     # Normalize using Min-Max scaling (0-1 values)
    x_norm = scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten() # Reshape into column vectors, transform data, flatten for convenience 
    y_norm = scaler.fit_transform(np.array(y).reshape(-1, 1)).flatten()
    z_norm = scaler.fit_transform(np.array(z).reshape(-1, 1)).flatten()
    vis_norm = scaler.fit_transform(np.array(vis).reshape(-1, 1)).flatten()
    return x_norm, y_norm, z_norm, vis_norm

# Create a single feature vector so we can measure euclidean distance between photos
def create_feature_vector(landmarks):
    x_norm, y_norm, z_norm, vis_norm = normalize_landmarks(landmarks)
    feature_vector = np.concatenate((x_norm, y_norm, z_norm, vis_norm))
    return feature_vector

def calculate_distance(feature_vector1, feature_vector2):
    return euclidean(feature_vector1, feature_vector2)



# Define the path to the archive folder
archive_path = 'archive/'

# PHOTO TO TEST: 
test_1 = 'archive/deadlift/deadlift_100031.jpg'
test_1 = 'archive/lat_pulldown/lat_pulldown_g9.jpg'
test_1 = 'archive/bench_press/bench_press_g18.jpg'
      

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
                if counter >= 2000:
                    break
                
                photo_path = os.path.join(root, file_name)

                if "bench_press_g18.jpg" in photo_path:
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


