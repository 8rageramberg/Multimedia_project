import numpy as np
import sys
import os
import numpy as np
import pandas as pd
import pandas as pd
from scipy.spatial import distance

class FeatureWeightOptimizer:
    """
    A class to optimize the weights of different features extracted by the feature_extractor.
    
    Attributes:
        feature_extractor (feature_extractor): The feature extractor instance with different feature extractors.
        data (pd.DataFrame): DataFrame containing features and labels.
        n_splits (int): Number of folds for cross-validation.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def calculate_score(self, path_1, path_2, w1, w2, w3):
        """
        Calculate the score between two images.

        Args:
            full_path (str): The path to the first image.
            other_full_path (str): The path to the second image.

        Returns:
            float: The score between the two images.
        """
        #compute pose estimator score1
        
        data = pd.read_csv("pose_estimator_features.pkl") 

        # Search for the rows corresponding to the image paths
        row_1 = data[data['photo_path'] == path_1]
        row_2 = data[data['photo_path'] == path_2]

        # Extract the feature vectors (assuming they are stored as strings)
        features_1 = row_1['features']  # Remove brackets and split into numbers
        features_2 = row_2['features']

        # Convert feature vectors to numeric format
        # features_1 = [float(x) for x in features_1]
        # features_2 = [float(x) for x in features_2]

        
        if features_1 is None or features_2 is None: 
            return 0
        
        # Loop keypoints and do euclidean distance for each row
        euclidean_dist = []
        for kp1, kp2 in zip(keypoints1, keypoints2):
            euclidean_dist.append(euclidean(kp1, kp2))

        # Flip so lower distances is higher similarity, then scale the similarity
        # score on a 1-100 range, and retrieve the mean score
        euclidean_dist_reshape = np.array(euclidean_dist).reshape(-1, 1)
       
        mean_eculidean = 1 - np.mean(euclidean_dist_reshape)/2.4494897428   
        mean_similarity_score = 100*(mean_eculidean)

        #similarity_scores = 100 - scaled_dist
        score1 = mean_similarity_score 

        #TODO: compute distance 2 and 3
        data = pd.read_csv("SIFT_descriptor_detector_features.pkl") 

        # Search for the rows corresponding to the image paths
        row_1 = data[data['photo_path'] == path_1]
        row_2 = data[data['photo_path'] == path_2]

        # Extract the feature vectors (assuming they are stored as strings)
        features_1 = row_1['features']  # Remove brackets and split into numbers
        features_2 = row_2['features']

        # Convert feature vectors to numeric format
        # features_1 = [float(x) for x in features_1]
        # features_2 = [float(x) for x in features_2]

        # Calculate the SIFT score2
        # Utilizing the cv2 libraries brute force matcher
        brute_force_matcher = cv.BFMatcher()
        matches = brute_force_matcher.knnMatch(features_1, features_2, 2)

        # Applying a ratio test (from: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
        good_matches = []
        good_match_ratio = 0.75	
        for neighbor_1, neighbor_2 in matches:
            if (neighbor_1.distance < good_match_ratio * neighbor_2.distance):
                good_matches.append([neighbor_1])

        # Calculate metric:
        num_good_matches = len(good_matches)
        if num_good_matches == 0:
            score2 = 0
        else:
            score2 = (num_good_matches / self.nr_descriptors) * 100



        data = pd.read_csv("CNN_features.pkl")  # Replace 'your_data.csv' with your actual file name

        # Search for the rows corresponding to the image paths
        row_1 = data[data['photo_path'] == path_1]
        row_2 = data[data['photo_path'] == path_2]

        # Extract the feature vectors (assuming they are stored as strings)
        features_1 = row_1['features'] 
        features_2 = row_2['features']

        # Convert feature vectors to numeric format
        # features_1 = [float(x) for x in features_1]
        # features_2 = [float(x) for x in features_2]

        # Calculate the CNN score3        
        distance = np.linalg.norm((self.features - features_to_compare) / 2500 * 100)
        score3 =  100 - (distance / 5000) * 100
        


        # Compute weighted sum of euclidean distances
        weighted_score = w1 * score1 + w2 * score2 + w3 * score3

        return weighted_score
    
    def extract_subfolder(self, path):
        parts = path.split('/')  # Split the path into segments
        if len(parts) > 1:
            return parts[-1]  # Return the second segment, which is the exercise subfolder
        return None
        

    def optimize_weights(self):
        """
        Optimizes the weights of each feature extractor to maximize classification accuracy.

        Returns:
            dict: Optimized weights for each feature extractor.
        """
        # Initialize weights
        weights = {extractor.get_name(): 1.0 for extractor in self.feature_extractor.list_of_extractors}        
        
        best_accuracy = 0
        best_weights = weights.copy()
        
        sys.path.append("feature_extraction")

        # Generate new weights combinations
        for _ in range(100):  # Perform 100 random adjustments
            test_weights = {k: np.random.rand() for k in weights}
            
            # for this weight assignment, count the number of images classified correctly, i.e. is the closest image in the same exercise folder?
            accuracy_count = 0 
            
            #iterate over all images, call compare function and accuracy_count += 1 if classified correctly
            archive_path = os.path.join(self.base_path, '..', 'archive')  # Go up one level and then to the archive folder
            for subdir, dirs, files in os.walk(archive_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                        relative_path = os.path.join('archive', os.path.relpath(os.path.join(subdir, file), archive_path))  # Get relative path
                        # TODO: find closest image among the other image, i.e. exclude the image itself in the comparison
                        closest_image = ""
                        best_score = float('inf')
                        for other_subdir, other_dirs, other_files in os.walk(archive_path):
                            for other_file in other_files:
                                if other_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                                    other_relative_path = os.path.join('archive', os.path.relpath(os.path.join(other_subdir, other_file), archive_path))
                                    if other_relative_path != relative_path:  # Exclude the image itself
                                        # TODO: Calculate score between images                                        
                                        score = self.calculate_score(relative_path, other_relative_path, test_weights[0], test_weights[1], test_weights[2])
                                        if score > best_score:
                                            best_score = score
                                            closest_image = other_relative_path

                        if closest_image != "":
                            # Check if closest image is in the same subdir
                            if self.extract_subfolder(closest_image) == self.extract_subfolder(relative_path):
                                accuracy_count += 1
                                
            if accuracy_count > best_accuracy:
                best_accuracy = accuracy_count
                best_weights = test_weights.copy()       
        
        print('best_weights are ' + str(best_weights))
        
        return best_weights