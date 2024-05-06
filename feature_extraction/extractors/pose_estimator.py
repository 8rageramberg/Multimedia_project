import mediapipe as mp
import cv2

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

# pip install mediapipe==0.10.9 
# wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
# MAC: curl -o pose_landmarker.task -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task


# https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
'''
Class representing pose estimation.

Attributes:
    - photo_path (Str): path of current photo
'''
class pose_estimator:
    name = "pose estimator"
    photo_path = None 

    def __init__(self, model_path = 'pose_landmarker_heavy'):
        self.model_path = model_path
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,         # individual image
            model_complexity=1,             # accuracy (how advanced, 0, 1 or 2)
            enable_segmentation=False,      # predict segmentation mask
            smooth_segmentation=False,      # filter segmentation across input to reduce jitter
            min_detection_confidence=0.5,   # Minimum confidence for pose detection for "success"
            min_tracking_confidence=0.5     # Minimum confidence for pose tracking to be "success"
        )
        
    def get_features(self, photo_path, display=False):
        self.photo_path = photo_path
        # Read input image and cast to RGB
        image = cv2.imread(self.photo_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and get pose landmarks
        results = self.pose.process(image_rgb)
        landmarks = results.pose_landmarks
    
        # Visualize the landmarks on the image
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
            return annotated_image, landmarks, keypoints
        else:
            print("something wrong with landmarks")
            return None, None, None
        
    def get_name(self):
        return "Pose Estimation"