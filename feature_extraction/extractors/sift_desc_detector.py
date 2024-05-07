from sklearn.decomposition import PCA
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


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
        
        

    def get_features(self, nr_descriptors=50):
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
        key_points, descriptors = sift.detectAndCompute(gray_curr_photo, None)


        # Sort the keypoints by "response strength" to only retrieve the
        # nr_descriptors descriptors of the keypoints with the highest response
        # strength if image contains nr_descriptors:
        if len(key_points) >= nr_descriptors:
            points_to_retrieve = nr_descriptors
        else:
            points_to_retrieve = len(key_points)

        responses = np.array([kp.response for kp in key_points])
        top_indices = np.argsort(-responses)[:points_to_retrieve]
        top_descriptors = np.array([descriptors[index] for index in top_indices])

        return top_descriptors
        
    
    def compare(self, descriptors_to_compare):
        return None


    def get_name(self):
        '''Getter function for getting extractor name'''
        return self.name
    


























# Curr references used:
# - COMP4425 lecture: https://canvas.sydney.edu.au/courses/56273/pages/week-08-large-scale-image-retrieval?module_item_id=2240287
# - OpenCV doc: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
# - PCA sift: https://ieeexplore.ieee.org/document/1315206
# - Cosine similarity: https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
# - OPENCV Bruteforce: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

# Unused:
    # def perform_PCA(self, descriptors):
    #     '''
    #     Function for performing PCA on the descriptors at hand.

    #     Parameters:
    #     - self (sift_desc_detector obj): Itself
    #     - descriptors (np.array): Array containing the top 40 descriptors

    #     Returns:
    #     - pca (PCA object): Sklearn PCA object used during transform
    #     - dim_reduced_descriptors(np.array): Dimensionality reduced array
    #     '''

    #     # Set PCA object to a secure random state and fit to the
    #     # number of objects.
    #     pca = PCA(random_state=22)
    #     pca.fit(descriptors)

    #     # Get the cumulative variance explained by each principal component:
    #     list_of_cumulative_variance = [sum(pca.explained_variance_ratio_[:i+1]) for i in range(len(pca.explained_variance_ratio_))]
    #     index_of_first_number_higher_than_90_percent = np.argmax(np.array(list_of_cumulative_variance) > 0.9)

    #     # Set the number of dimensions based on the cumulative variance explained:
    #     pca = PCA(n_components=index_of_first_number_higher_than_90_percent, random_state=22)
    #     dim_reduced_descriptors = pca.fit_transform(descriptors)

    #     return pca, dim_reduced_descriptors