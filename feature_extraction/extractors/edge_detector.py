'''
Class representing an edge detector.

Attributes:
    - photo_path (Str): path of current photo
'''
class edge_detector:
    name = "Edge Detector"
    photo_path = None

    def __init__(self, photo_path=""):
        self.photo_path = photo_path
        
    '''
    Main func:
    '''
    def get_features(self):
        # TODO: Implement a feature extraction algo.
        # For example: with open(self.photo_path): *some math*

        return None
    
    '''Getter function for getting extractor name'''
    def get_name(self):
        return self.name