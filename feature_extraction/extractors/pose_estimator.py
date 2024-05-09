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

class pose_estimator:
    '''
    Class representing a Pose estimator. The Pose estimator classes
    utilizes google's MediaPipe framework to compose a pose from a
    input image and features to retrieve this and compare. 

    Attributes:
        - photo_path (Str): Path of current photo
        - model_path (Str)

    Functions:
        - get_features(): Retrieve the pose features

        - compare(): Compare to poses against eachother
                    and retrieve a mean similarity score.

    '''
    name = "Pose_estimator"
    photo_path = None
    own_keypoints = None
    
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
        '''
        Function to retrieve features in the sense of keypoints
        to the pose of the image currently attached. 

        Parameters:
            - display (Boolean):              To display the features after retrieval

        Returns:
            - keypoints (List[keypoint obj]): A list of keypoint objects representing
                                              the current image pose
        '''
        # Read input image and project onto other color space
        image = cv2.imread(self.photo_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

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
            
            # How to extract the coordinates (keypoints) from a landmark object
            # martiniblack: https://github.com/google/mediapipe/issues/1020
            keypoints = []                                              
            for feature in results.pose_landmarks.landmark:
                keypoints.append({
                                    'X': feature.x,
                                    'Y': feature.y,
                                    'Z': feature.z,
                                    'Visibility': feature.visibility,
                                    })
            self.own_keypoints = keypoints
            return keypoints
        else:
            #print("Something wrong with landmarks, features not retrieved")
            return None
    
    

    def compare(self, keypoints_to_compare):
        '''
        Function for comparing own keypoints agains other keypoint lists

        Parameters:
            - keypoints_to_compare (List[keypoints obj]): List of keypoints to compare against
                                                          it's own.

        Returns:
            - mean_similarity_score (Float[0-100]):       A mean similarity score float
        '''
        # Error handeling, if landmarks on image did not exist None is returned,
        # if None, there is a 0% match
        if self.own_keypoints is None or keypoints_to_compare is None: 
            return 0
        
        # Loop keypoints and do euclidean distance for each row
        euclidean_dist = []
        for kp1, kp2 in zip(self.own_keypoints, keypoints_to_compare):
            features_1 = np.array([kp1['X'], kp1['Y'], kp1['Z'], kp1['Visibility']])
            features_2 = np.array([kp2['X'], kp2['Y'], kp2['Z'], kp2['Visibility']])
            euclidean_dist.append(euclidean(features_1, features_2))
        
        # Flip so lower distances is higher similarity, then scale the similarity
        # score on a 1-100 range, and retrieve the mean score
        euclidean_dist_reshape = np.array(euclidean_dist).reshape(-1, 1)
        scaled_dist = self.scaler.fit_transform(euclidean_dist_reshape)
        similarity_scores = 1 - scaled_dist
        scaled_similarity_scores = similarity_scores * 100
        mean_similarity_score = np.mean(scaled_similarity_scores)   
        return mean_similarity_score
        


    def get_name(self):
        '''Getter function for getting extractor name'''
        return self.name
    
    def set_new_photo(self, photo_path):
        '''Setter function for setting a new photo'''
        self.photo_path = photo_path
