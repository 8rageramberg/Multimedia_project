from sklearn.decomposition import PCA
import numpy as np
import cv2 as cv


class sift_desc_detector:
    '''
    Class representing a SIFT (Scale invariant feature transform)
    descriptor detector which automatically detects the 20 most
    representative descriptors via PCA (Principal component analysis)
    and returns the as a list of vectors.

    Attributes:
        - photo_path (Str): path of current photo
    '''

    name = "SIFT descriptor detector"
    photo_path = None

    def __init__(self, photo_path=""):
        self.photo_path = photo_path
        
        

    def get_features(self):
        '''
        This get_features function retrieves low dimensional
        sift descriptors of the current image:

        Parameters:
        - self (sift_desc_detector obj): Itself

        Returns
        - descriptors (list[descriptors]): List of descriptors of curr img
        '''
        # Retrieve the current photo and turn it to greyscale
        # to minimize color space dimensionality and effectivize
        # processing
        curr_photo = cv.imread(self.photo_path)
        gray_curr_photo = cv.cvtColor(curr_photo, cv.COLOR_BGR2GRAY)

        # Initialize sift object and utilize on the greyscale
        # of our image to retrieve descriptors:
        sift = cv.SIFT.create()
        
        # TODO: Implementer PCA + 
        key_points, descriptors = sift.detectAndCompute(gray_curr_photo, None)

        return descriptors
    

    def get_name(self):
        '''Getter function for getting extractor name'''
        return self.name
    






# Curr references used:
# - COMP4425 lecture: https://canvas.sydney.edu.au/courses/56273/pages/week-08-large-scale-image-retrieval?module_item_id=2240287
# - OpenCV doc: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
# - PCA sift: https://ieeexplore.ieee.org/document/1315206