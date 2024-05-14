import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import sys
import os
import random

import mediapipe as mp
import cv2 as cv
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
import pandas as pd
import colorama
from colorama import Fore
import math

sys.path.append("feature_db")

class FeatureWeightOptimizer:
    """
    A class to optimize the weights of different features extracted by the feature_extractor.
    
    
    """
    def normalize_tuple(t):
        """Normalize a tuple by dividing each element by the GCD of the tuple."""
        gcd = math.gcd(t[0], math.gcd(t[1], t[2]))
        return tuple(x // gcd for x in t)

    def _read_dbs(self):
        ''' Private function for reading the DB'''
        pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_db")

        db_1 = pd.read_pickle(os.path.join(pkl_path, "SIFT_descriptor_detector_features.pkl"))
        db_2 = pd.read_pickle(os.path.join(pkl_path, "Pose_estimator_features.pkl"))
        db_3 = pd.read_pickle(os.path.join(pkl_path, "CNN_features.pkl"))
        
        return db_1, db_2, db_3
    

        
    
    def calculate_score(self, path_1, path_2, weights, pose_1, sift_1, cnn_1, pose_2, sift_2, cnn_2):
        # Rest of the code...

        """

        Args:
            full_path (str): The path to the first image.
            other_full_path (str): The path to the second image.

        Returns:
            float: The score between the two images.
        """
        #compute pose estimator score1
        

        # Extract the feature vectors (assuming they are stored as strings)
        features_1 = [pose_1, sift_1, cnn_1] 
        features_2 = [pose_2, sift_2, cnn_2]    

        
        if features_1[0] is None or features_2[0] is None: 
            score1 = 0
        else:        
            # Loop keypoints and do euclidean distance for each row
            euclidean_dist = []
            for kp1, kp2 in zip(features_1[0], features_2[0]):
                euclidean_dist.append(euclidean(kp1, kp2))
            # Flip so lower distances is higher similarity, then scale the similarity
            # score on a 1-100 range, and retrieve the mean score
            euclidean_dist_reshape = np.array(euclidean_dist).reshape(-1, 1)
        
            mean_eculidean = 1 - np.mean(euclidean_dist_reshape)/2.4494897428   
            mean_similarity_score = 100*(mean_eculidean)

            #similarity_scores = 100 - scaled_dist
            score1 = mean_similarity_score 

        # Calculate the SIFT score2     

        # Utilizing the cv2 libraries brute force matcher
        brute_force_matcher = cv.BFMatcher()
        matches = brute_force_matcher.knnMatch(features_1[1], features_2[1], 2)

        # Applying a ratio test (from: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
        good_matches = []
        good_match_ratio = 0.75	
        for neighbor_1, neighbor_2 in matches:
            if (neighbor_1.distance < good_match_ratio * neighbor_2.distance):
                good_matches.append([neighbor_1])

        # Calculate metric:
        num_good_matches = len(good_matches)
        nr_descriptors = 50 
        if num_good_matches == 0:
            score2 = 0
        else:
            score2 = (num_good_matches / nr_descriptors) * 100

        # Calculate the CNN score3        
        distance = np.linalg.norm((features_1[2] - features_2[2]))
        score3 =  100 - (distance / 5000) * 100
        # Compute weighted sum of euclidean distances
        weighted_score = weights[0] * score1 + weights[1] * score2 + weights[2] * score3

        return weighted_score
    
    def extract_subfolder(self, path):
        parts = path.split('/')  # Split the path into segments
        if len(parts) > 1:
            return parts[-2]  # Return the second segment, which is the exercise subfolder
        return None
    
    def find_path(self, image):
        for root, _, files in os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive")): 
            if image in files:
               return os.path.join(root, image)
        

    def optimize_weights(self):
        """
        Optimizes the weights of each feature extractor to maximize classification accuracy.

        Returns:
            dict: Optimized weights for each feature extractor.
        """
        # Initialize weights
        list_of_extractors = ["pose_estimator", "SIFT_descriptor_detector", "CNN"]
        weights = [1, 1, 1]    
        
        best_accuracy = 0
        best_weights = weights.copy()
        
        sys.path.append("feature_extraction")

        db_1, db_2, db_3 = self._read_dbs()

        image_files = []
        archive_path = 'archive'
        for subdir, dirs, files in os.walk(archive_path):  
            if 'A_test_set' in dirs:
                dirs.remove('A_test_set')                    
            if 'Test_plank' in dirs:
                dirs.remove('Test_plank')      
            if 'Test_plank' in dirs:
                dirs.remove('Test_pullup') 
            if 'tester_imgs' in dirs:
                dirs.remove('tester_imgs')                     
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(subdir, file)
                    if "DS_Store" in full_path:
                        continue
                    image_files.append(full_path)

        s = {2, 4, 6, 8, 10}
        assignments = {self.normalize_tuple([x, y, z]) for x in s for y in s for z in s}
        print(assignments)
        # Generate new weights combinations
        for test_weights in assignments:  # Perform 100 random adjustments
            print(Fore.RED + f'Testing weights: {test_weights}' + Fore.RESET)
                        
            # for this weight assignment, count the number of images classified correctly, i.e. is the closest image in the same exercise folder?
            accuracy_count = 0 
            #new iteration
            subset_size = 10
            random_subset = random.sample(image_files, subset_size)         

            for first_path in random_subset:
                closest_image = ""
                best_score = float('-inf')

                #find the feature vectors for the first image
                pose_feature1 = db_2[db_2['Filename'] == os.path.basename(first_path)]['Feature'].iloc[0]
                sift_feature1 = db_1[db_1['Filename'] == os.path.basename(first_path)]['Feature'].iloc[0]
                cnn_feature1 = db_3[db_3['Filename'] == os.path.basename(first_path)]['Feature'].iloc[0]

                for i in db_1.index:
                    if db_1['Filename'][i] == os.path.basename(first_path):
                        continue # Skip the image itself
                    pose_feature2, sift_feature2, cnn_feature3 = db_2['Feature'][i], db_1['Feature'][i], db_3['Feature'][i]
                    score = self.calculate_score(os.path.basename(first_path), db_1['Filename'][i], test_weights, pose_feature1, sift_feature1, cnn_feature1, pose_feature2, sift_feature2, cnn_feature3)
                    if score > best_score:
                        best_score = score
                        closest_image = db_1['Filename'][i]
                if closest_image != "":
                    # Check if closest image is in the same subdir
                    if self.extract_subfolder(self.find_path(closest_image)) == self.extract_subfolder(first_path):
                        accuracy_count += 1
            print(f'Accuracy count: {accuracy_count} out of {subset_size} images.')
            if accuracy_count > best_accuracy:
                best_accuracy = accuracy_count
                best_weights = test_weights.copy() 
            print(f'current best weights are: {best_weights}')      

        print('best_weights are ' + str(best_weights))        
        return best_weights
    
if __name__ == "__main__":
    optimizer = FeatureWeightOptimizer()
    
    # Set the base path for the image archive
    
    # Optimize weights
    optimized_weights = optimizer.optimize_weights()
    print(optimized_weights)
