import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

class cnn:
    name = "CNN"
    photo_path = None

    def __init__(self, photo_path=""):
        self.photo_path = photo_path
        self.model = VGG16(weights='imagenet', include_top=False)  # Load VGG16 without the classification layers

    def get_features(self):
        img = image.load_img(self.photo_path, target_size=(224, 224))  # VGG16 input size
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # Preprocess the image for VGG16

        features = self.model.predict(x)  # Extract features using VGG16
        features = features.flatten()  # Flatten the features into a vector

        return features
    
    def compare(self, descriptors_to_compare):
        return np.linalg.norm(self.get_features() - descriptors_to_compare)

    def get_name(self):
        return self.name