'''
Class representing an Extractor 2.

Attributes:
    - photo_url (Str): URL of current photo
'''
class extractor_2:
    name = "Extractor 2"
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
    
    '''Getter function for getting extractor name'''
    def get_name(self):
        return self.name