import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import sys
import os

class cnn:
    name = "CNN"
    photo_path = None
    features = None

    def __init__(self, photo_path="", weights_path="/Users/brageramberg/Desktop/Multimedia_project/feature_extraction/extractors/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"):
        self.photo_path = photo_path
        self.model = VGG16(weights=None, include_top=False)  # Load VGG16 without the classification layers
        self.model.load_weights(weights_path)

    def get_features(self):
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
        if self.features is None or features_to_compare is None: 
            return 0
        distance = np.linalg.norm((self.features - features_to_compare) / 2500 * 100)
        return 100 - (distance / 5000) * 100

    def get_name(self):
        return self.name

    def set_new_photo(self, photo_path):
        '''Setter function for setting a new photo'''
        self.photo_path = photo_path