import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import sys
import os

class cnn:
    '''
    A class representing a CNN extractor. The CNN extractor extracts a NumPy array
    of features with tensorflows VGG16 object setting the weights to "imagenet".

    Attributes:
        - name (Str):                 Str representation of the extractors name.
        - photo_path (Str):           Path to the current photo used by extractor.
        - own_keypoints (np.ndarray): Own CNN features retrieved.

    Functions:
        - get_features():  Retrieve the CNN features of the current photo.
        - compare():       Compare to CNN features against eachother and retrieve a mean
                           similarity score.
        - get_name():      Retrieves the name of the extractors. 
        - set_new_photo(): Sets a new photo to the extractor.
    '''
    name = "CNN"
    photo_path = None
    features = None

    def _init_(self, photo_path="", weights_path=""):
        if not weights_path:
            weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
        self.photo_path = photo_path
        self.model = VGG16(weights=None, include_top=False)  # Load VGG16 without the classification layers
        self.model.load_weights(weights_path)


    def get_features(self):
        '''
        Function for extracting CNN features from the specific image
        set at init of class.

        Returns:
            - features (np.ndarray): A NumPy array containing the CNN
                                     features of the current img.
        '''
        img = image.load_img(self.photo_path, target_size=(224, 224))  # VGG16 input size
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # Preprocess the image for VGG16

        # Stop standard output and standard error
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        features = self.model.predict(x)  # Extract features using VGG16
        features = features.flatten()  # Flatten the features into a vector

        # Restore standard output and standard error
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        self.features = features
        return features
    

    def compare(self, features_to_compare):
        '''
        Function for comparing own CNN feature vector against
        another feature vector.

        Parameters:
            - features_to_compare (np.ndarray): NumPy array of features
                                                to compare against own
        Returns:
            - compare_output (float):           A compare output between
                                                0 - 100
         '''
        if self.features is None or features_to_compare is None: 
            return 0
        distance = np.linalg.norm((self.features - features_to_compare) / 2500 * 100)
        return 100 - (distance / 5000) * 100



    def get_name(self):
        '''Getter function for getting name of the class'''
        return self.name

    def set_new_photo(self, photo_path):
        '''Setter function for setting a new photo'''
        self.photo_path = photo_path