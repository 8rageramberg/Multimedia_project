from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#@title Licensed under the Apache License, Version 2.0 (the "License");
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


#wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
# MAC: curl -o pose_landmarker.task -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

'''
Class representing pose estimation.

Attributes:
    - photo_path (Str): path of current photo
'''
class pose_estimation:
    name = "Extractor 2"
    photo_path = None

    def __init__(self, photo_path=""):
        self.photo_path = photo_path
        
    '''
    Main func:
    '''
    def get_features(self, display=False):
        # TODO: Implement a feature extraction algo.
        # For example: with open(self.photo_path): *some math*
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        detector = vision.PoseLandmarker.create_from_options(options)

        # STEP 3: Load the input image.
        image = mp.Image.create_from_file(self.photo_path)

        # STEP 4: Detect pose landmarks from the input image.
        detection_result = detector.detect(image)

        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)
        BGR_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255

        if display: 
            image = cv2.imread(self.photo_path, 0)
            cv2.imshow("Photo", image)
            cv2.waitKey(0)  # Press a key to remove the window
            cv2.destroyAllWindows()

            cv2.imshow("BGR Annotated", BGR_annotated_image)
            cv2.waitKey(0)  # Press a key to remove the window
            cv2.destroyAllWindows()

            cv2.imshow("kukk", visualized_mask)
            cv2.waitKey(0)  # Press a key to remove the window
            cv2.destroyAllWindows()

        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
        return None
    
    '''Getter function for getting extractor name'''
    def get_name(self):
        return self.name
    

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)
        bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Loop through the detected poses to visualize. 
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            bgr_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        return bgr_image
    

test_1 = '/Users/brageramberg/Desktop/Multimedia_Project/archive/hip thrust/hip thrust_100081.jpg'
shit = pose_estimation(test_1)
shit.get_features(display=True)