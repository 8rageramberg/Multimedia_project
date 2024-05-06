import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
import sys
sys.path.append("feature_extraction")
from feature_extraction.extractors.pose_estimator import pose_estimator

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


test_1 = 'archive/A_test_set/Test_plank/plank_704601_copy.jpg'
test_2 = 'archive/A_test_set/Test_plank/plank_g8_copy.jpg'
test_3 = 'archive/shoulder_press/shoulder_press_100121.jpg' 

pose_estimator = pose_estimator(test_1)
annotated_image_1, landmarks_1, keypoints_1 = pose_estimator.get_features(display=True)

pose_estimator_2 = pose_estimator(test_2)
annotated_image_2, landmarks_2, keypoints_2 = pose_estimator_2.get_features(display=True)

pose_estimator_3 = pose_estimator(test_3)
annotated_image_3, landmarks_3, keypoints_3 = pose_estimator_3.get_features(display=True)

# Create feature vectors for each set of landmarks
feature_vector1 = create_feature_vector(keypoints_1)
feature_vector2 = create_feature_vector(keypoints_2)
feature_vector3 = create_feature_vector(keypoints_3)

# Compare normalized feature vectors
distance = calculate_distance(feature_vector1, feature_vector2)
print("Distance between feature vectors 1 and 2 :", distance)


distance = calculate_distance(feature_vector2, feature_vector3)
print("Distance between feature vectors 2 and 3 :", distance)

distance = calculate_distance(feature_vector1, feature_vector3)
print("Distance between feature vectors 1 and 3 :", distance)