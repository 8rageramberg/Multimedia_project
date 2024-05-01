from .extractors.edge_detector import edge_detector
from .extractors.extractor_1 import extractor_1
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

    edge_detector = None
    extractor_1 = None
    extractor_2 = None

    list_of_extractors = []


    # Initiate the feature extractor
    def __init__(self, photo_path=""):
        # Set the path of the photo
        self.photo_path = photo_path

        # TODO: Add more features
        self.edge_detector = edge_detector(photo_path)
        self.extractor_1 = extractor_1(photo_path)
        self.extractor_2 = extractor_2(photo_path)

        # TODO: For every feature added to list of feature extractors
        self.list_of_feature_extractors.append(self.edge_detector)
        self.list_of_feature_extractors.append(self.extractor_1)
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


    '''Setter function for setting photo path'''
    def set_photo_path(self, new_path):
        self.photo_path = new_path


    '''Getter function for getting photo path'''
    def get_photo_path(self):
        return self.photo_path