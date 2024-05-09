import mediapipe as mp
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean

# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
# not sure if you need more than mediapipe, but keep in case
# wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
# MAC: curl -o pose_landmarker.task -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

# https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
'''
Class representing pose estimation.

Attributes:
    - photo_path (Str): path of current photo
'''
class pose_estimator:
    name = "Pose_estimator"
    photo_path = None 
    
    def __init__(self, photo_path="", model_path = 'pose_landmarker_heavy'):
        self.model_path = model_path
        self.photo_path = photo_path

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,         # individual image
            model_complexity=2,             # accuracy (how advanced, 0, 1 or 2)
            enable_segmentation=False,      # predict segmentation mask
            smooth_segmentation=False,      # filter segmentation across input to reduce jitter
            min_detection_confidence=0.5,   # Minimum confidence for pose detection for "success"
            min_tracking_confidence=0.5     # Minimum confidence for pose tracking to be "success"
        )

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def get_features(self, display=False):
        image = cv2.imread(self.photo_path)                 # Read input image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Cast to RGB

        # Process the image and get pose landmarks
        results = self.pose.process(image_rgb)
        landmarks = results.pose_landmarks
    
        # landmarks on the image
        if landmarks:
            annotated_image = image.copy()
            self.mp_drawing.draw_landmarks(annotated_image, landmarks, self.mp_pose.POSE_CONNECTIONS)
         
            if display:
                cv2.imshow("Annotated Image", annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            keypoints = []                                              # martiniblack: https://github.com/google/mediapipe/issues/1020
            for feature in results.pose_landmarks.landmark:             # How to extract the coordinates (keypoints) from a landmark object
                keypoints.append({
                                    'X': feature.x,
                                    'Y': feature.y,
                                    'Z': feature.z,
                                    'Visibility': feature.visibility,
                                    })
            return keypoints
        else:
            print("Something wrong with landmarks, features not retrieved")
            return None
    
    # Compare compares two keypoint lists
    def compare(self, keypoints_1, keypoints_2):
        if keypoints_1 is None or keypoints_2 is None:              # Error handeling, if landmarks on image did not exist None is returned, 
            return 0                                                # if None, there is a 0% match
        
        euclidean_dist = []
        for kp1, kp2 in zip(keypoints_1, keypoints_2):              # Loop keypoints and do euclidean distance for each row
            features_1 = np.array([kp1['X'], kp1['Y'], kp1['Z'], kp1['Visibility']])
            features_2 = np.array([kp2['X'], kp2['Y'], kp2['Z'], kp2['Visibility']])
            euclidean_dist.append(euclidean(features_1, features_2))
        
        euclidean_dist_reshape = np.array(euclidean_dist).reshape(-1, 1)
        scaled_dist = self.scaler.fit_transform(euclidean_dist_reshape)
        similarity_scores = 1 - scaled_dist                         # Flip so lower distances is higher similarity
        scaled_similarity_scores = similarity_scores * 100          # Scale the similarity scores to the 1-100 range
        mean_similarity_score = np.mean(scaled_similarity_scores)   # Mean of scores
        return mean_similarity_score
        
    def get_name(self):
        '''Getter function for getting extractor name'''
        return self.name
    
    def set_new_photo(self, photo_path):
        '''Setter function for setting a new photo'''
        self.photo_path = photo_path
