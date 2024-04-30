'''
Class representing an edge detector.

Attributes:
    - photo_url (Str): URL of current photo
'''
class edge_detector:
    name = "Edge Detector"
    photo_url = None

    def __init__(self, photo_url=""):
        self.photo_url = photo_url
        
    '''
    Main func:
    '''
    def get_features(self):
        # TODO: Implement a feature extraction algo.
        # For example: with open(self.photo_url): *some math*

        return None
    
    '''Getter function for getting photo url'''
    def get_name(self):
        return self.name