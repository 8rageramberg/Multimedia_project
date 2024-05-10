import numpy as np
import sys
import os
import numpy as np


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
                        full_path = os.path.join(subdir, file)     
                    #TODO: find closest image

                    closest_image = ""
                    if closest_image == subdir:
                        accuracy_count += 1
            

                
                
                
            if accuracy_count > best_accuracy:
                best_accuracy = accuracy_count
                best_weights = test_weights.copy()

        return best_weights



