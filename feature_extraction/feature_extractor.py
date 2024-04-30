from .extractors.edge_detector import edge_detector


'''
A class reperesenting a feature extractor. The feature extractor has
different extractor objects for specific extracting purposes.

Attributes:
    - photo_url (Str): URL of current photo
    - edge_detector (edge_detecor): Edge detector object
    - list_of_extractor (List[extractors]): List of all extractors

Functions:
    - extract():              Function for extracting each feature of an image
                              utilizing all extractors.

    - set_photo_url(new_url): Function for setting the photo_url

    - get_photo_url():        Function for retrieving the photo_url
'''
class feature_extractor:
    # Class variables for storing photo url, feature_extractors
    # and all extractors in a list for future use.
    photo_url = None
    edge_detector = None
    list_of_extractors = []


    # Initiate the feature extractor
    def __init__(self, photo_url=""):
        # Set the url of the photo
        self.photo_url = photo_url

        # TODO: Add more features
        self.edge_detector = edge_detector(photo_url)

        # TODO: For every feature added to list of feature extractors
        self.list_of_feature_extractors.append(self.edge_detector)


    '''
    Main function for retrieving features.
     
    Returns:
        - features (List[(Str, List[features])]): A list of tuples containing extractor name and featues
    '''
    def extract(self):
        # Extract features and return
        features = [(extractor.get_name(), extractor.get_features()) for extractor in self.list_of_feature_extractors]
        return features


    '''Setter function for setting photo url'''
    def set_photo_url(self, new_url):
        self.photo_url = new_url


    '''Getter function for getting photo url'''
    def get_photo_url(self):
        return self.photo_url