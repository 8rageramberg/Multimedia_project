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

    Functions:
        - get_features(n_descriptors):     A function for calculating the sift descriptors of the
                                           specified image. Returns a number of descriptors based on
                                           the param set at utilization. Default = 50.

        - compare(descriptors_to_compare): A function for comparing the features of the current
                                           image to features given as a parameter.
    '''

    name = "SIFT_descriptor_detector"
    photo_path = None
    nr_descriptors = None 
    good_match_ratio = None
    features = None

    def __init__(self, photo_path="", nr_descriptors=50, good_match_ratio = 0.75):
        self.photo_path = photo_path
        self.nr_descriptors = nr_descriptors
        self.good_match_ratio = good_match_ratio
        
        

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
        key_points, descriptors = sift.detectAndCompute(gray_curr_photo, None)


        # Sort the keypoints by "response strength" to only retrieve the
        # nr_descriptors descriptors of the keypoints with the highest response
        # strength if image contains nr_descriptors:
        if len(key_points) >= self.nr_descriptors:
            points_to_retrieve = self.nr_descriptors
        else:
            points_to_retrieve = len(key_points)

        responses = np.array([kp.response for kp in key_points])
        top_indices = np.argsort(-responses)[:points_to_retrieve]
        top_descriptors = np.array([descriptors[index] for index in top_indices])

        self.features = top_descriptors

        return top_descriptors
        
    
    def compare(self, descriptors_to_compare):
        '''
        A compare function to compare the features (descriptors) of
        the current image to the features given as a param.

        Parameters:
            - descriptors_to_compare (nd.array): Array of descriptors

        Returns:
            - comparison_metric (float):         Metric evaluating comparison, float [0-100]
        '''
        # Utilizing the cv2 libraries brute force matcher
        brute_force_matcher = cv.BFMatcher()
        matches = brute_force_matcher.knnMatch(self.features, descriptors_to_compare, 2)

        # Applying a ratio test (from: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
        good_matches = []
        for neighbor_1, neighbor_2 in matches:
            if (neighbor_1.distance < self.good_match_ratio*neighbor_2.distance):
                good_matches.append([neighbor_1])

        # Calculate metric:
        num_good_matches = len(good_matches)
        if num_good_matches == 0:
            return 0
        else:
            return (num_good_matches / self.nr_descriptors)*100
        


    def get_name(self):
        '''Getter function for getting extractor name'''
        return self.name
    

    def set_new_photo(self, photo_path):
        '''Setter function for setting a new photo'''
        self.photo_path = photo_path
    


























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