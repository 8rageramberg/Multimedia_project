'''
Class representing a SIFT (Scale invariant feature transform)
descriptor detector which automatically detects the 20 most
representative descriptors via PCA (Principal component analysis)
and returns the as a list of vectors.

Attributes:
    - photo_path (Str): path of current photo
'''
class sift_desc_detector:
    name = "SIFT descriptor detector"
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