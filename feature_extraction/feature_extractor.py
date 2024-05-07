from .extractors.pose_estimator import pose_estimator
from .extractors.sift_desc_detector import sift_desc_detector
from .extractors.extractor_2 import extractor_2


'''
A class reperesenting a feature extractor. The feature extractor has
different extractor objects for specific extracting purposes.

Attributes:
    - photo_path (Str): path of current photo
    - edge_detector (edge_detecor): Edge detector object
    - list_of_extractor (List[extractors]): List of all extractors

Functions:
    - extract():              Function for extracting each feature of an image
                              utilizing all extractors.

    - set_photo_path(new_path): Function for setting the photo_path

    - get_photo_path():        Function for retrieving the photo_path
'''
class feature_extractor:
    # Class variables for storing photo path, feature_extractors
    # and all extractors in a list for future use.
    photo_path = None

    pose_estimator = None
    sift_desc_detector = None
    extractor_2 = None

    list_of_extractors = []


    # Initiate the feature extractor
    def __init__(self, photo_path=""):
        # Set the path of the photo
        self.photo_path = photo_path

        # TODO: Add more features
        self.pose_estimator = pose_estimator(photo_path)
        self.sift_desc_detector = sift_desc_detector(photo_path)
        self.extractor_2 = extractor_2(photo_path)

        # TODO: For every feature added to list of feature extractors
        self.list_of_feature_extractors.append(self.pose_estimator)
        self.list_of_feature_extractors.append(self.sift_desc_detector)
        self.list_of_feature_extractors.append(self.extractor_2)


    '''
    Main function for retrieving features.
     
    Returns:
        - features (List[(Str, List[features])]): A list of tuples containing extractor name and featues
    '''
    def extract(self):
        # Extract features and return
        features = [(extractor.get_name(), extractor.get_features()) for extractor in self.list_of_feature_extractors]
        return features


    def compare(self, image_to_compare_path):
        list_of_compare_outputs = [extractor.compare(image_to_compare_path) for extractor in self.list_of_feature_extractors]
        
        # TODO: Weight outputs for example (go through each output and put a weight):
        for index, extractor, output in enumerate(zip(self.list_of_feature_extractors, list_of_compare_outputs)):
            if (extractor.get_name() == "SIFT descriptor detector"):
                list_of_compare_outputs[index] = 0.2 * output #weighted output
            elif (extractor.get_name() == "Other extractor"):
                list_of_compare_outputs[index] = 0.3 * output #weighted output
            else:
                list_of_compare_outputs[index] = 0.4 * output #weighted output


    '''Setter function for setting photo path'''
    def set_photo_path(self, new_path):
        self.photo_path = new_path


    '''Getter function for getting photo path'''
    def get_photo_path(self):
        return self.photo_path