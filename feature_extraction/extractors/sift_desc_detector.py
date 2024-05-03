from sklearn.decomposition import PCA # type: ignore

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

        

        return None
    
    '''Getter function for getting extractor name'''
    def get_name(self):
        return self.name
    



# Curr references used:
# - COMP4425 lecture: https://canvas.sydney.edu.au/courses/56273/pages/week-08-large-scale-image-retrieval?module_item_id=2240287
# - OpenCV doc: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
# - PCA sift: https://ieeexplore.ieee.org/document/1315206