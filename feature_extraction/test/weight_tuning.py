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
    
    def calculate_distance(self, path_1, path_2, w1, w2, w3):
        """
        Calculate the distance between two images.

        Args:
            full_path (str): The path to the first image.
            other_full_path (str): The path to the second image.

        Returns:
            float: The distance between the two images.
        """
        #compute distance 1
        
        data = pd.read_csv("pose_estimator_features", sep="|") 

        # Search for the rows corresponding to the image paths
        row_1 = data[data['photo_path'] == path_1]
        row_2 = data[data['photo_path'] == path_2]

        # Extract the feature vectors (assuming they are stored as strings)
        features_1 = row_1['features'].iloc[0].strip('[]').split(', ')  # Remove brackets and split into numbers
        features_2 = row_2['features'].iloc[0].strip('[]').split(', ')

        # Convert feature vectors to numeric format
        features_1 = [float(x) for x in features_1]
        features_2 = [float(x) for x in features_2]

        # Calculate the Euclidean distance
        distance1 = distance.euclidean(features_1, features_2)        

        #TODO: compute distance 2 and 3
        data = pd.read_csv("SIFT_descriptor_detector_features", sep="|") 

        # Search for the rows corresponding to the image paths
        row_1 = data[data['photo_path'] == path_1]
        row_2 = data[data['photo_path'] == path_2]

        # Extract the feature vectors (assuming they are stored as strings)
        features_1 = row_1['features'].iloc[0].strip('[]').split(', ')  # Remove brackets and split into numbers
        features_2 = row_2['features'].iloc[0].strip('[]').split(', ')

        # Convert feature vectors to numeric format
        features_1 = [float(x) for x in features_1]
        features_2 = [float(x) for x in features_2]

        # Calculate the Euclidean distance
        distance2 = distance.euclidean(features_1, features_2)  

        data = pd.read_csv("pose_estimator_features", sep="|")  # Replace 'your_data.csv' with your actual file name

        # Search for the rows corresponding to the image paths
        row_1 = data[data['photo_path'] == path_1]
        row_2 = data[data['photo_path'] == path_2]

        # Extract the feature vectors (assuming they are stored as strings)
        features_1 = row_1['features'].iloc[0].strip('[]').split(', ')  # Remove brackets and split into numbers
        features_2 = row_2['features'].iloc[0].strip('[]').split(', ')

        # Convert feature vectors to numeric format
        features_1 = [float(x) for x in features_1]
        features_2 = [float(x) for x in features_2]

        # Calculate the Euclidean distance
        distance3 = distance.euclidean(features_1, features_2)  
        


        # Compute weighted sum of euclidean distances
        weighted_distance = w1 * distance1 + w2 * distance2 + w3 * distance3

        return weighted_distance
    
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
                        closest_distance = float('inf')
                        for other_subdir, other_dirs, other_files in os.walk(archive_path):
                            for other_file in other_files:
                                if other_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                                    other_relative_path = os.path.join('archive', os.path.relpath(os.path.join(other_subdir, other_file), archive_path))
                                    if other_relative_path != relative_path:  # Exclude the image itself
                                        # TODO: Calculate distance between images                                        
                                        distance = self.calculate_distance(relative_path, other_relative_path, test_weights[0], test_weights[1], test_weights[2])
                                        if distance < closest_distance:
                                            closest_distance = distance
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
    
    