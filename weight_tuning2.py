import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import sys
import os
import random
import time
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
    @staticmethod
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
        # Compute pose estimator score
        features_1 = [pose_1, sift_1, cnn_1]
        features_2 = [pose_2, sift_2, cnn_2]    

        if features_1[0] is None or features_2[0] is None: 
            score1 = 0
        else:        
            euclidean_dist = [euclidean(kp1, kp2) for kp1, kp2 in zip(features_1[0], features_2[0])]
            mean_eculidean = 1 - np.mean(np.array(euclidean_dist).reshape(-1, 1))/2.4494897428   
            score1 = 100 * mean_eculidean 

        # Calculate the SIFT score
        brute_force_matcher = cv.BFMatcher()
        matches = brute_force_matcher.knnMatch(features_1[1], features_2[1], 2)
        good_matches = [neighbor_1 for neighbor_1, neighbor_2 in matches if neighbor_1.distance < 0.75 * neighbor_2.distance]
        score2 = (len(good_matches) / 50) * 100 if good_matches else 0

        # Calculate the CNN score
        distance = np.linalg.norm(features_1[2] - features_2[2])
        score3 = 100 - (distance / 5000) * 100

        # Compute weighted sum of scores
        weighted_score = weights[0] * score1 + weights[1] * score2 + weights[2] * score3
        return weighted_score
    
    def extract_subfolder(self, path):
        parts = path.split('/')
        return parts[-2] if len(parts) > 1 else None
    
    def find_path(self, image):
        for root, _, files in os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive")): 
            if image in files:
               return os.path.join(root, image)
        
    def optimize_weights(self, subset_size):
        """
        Optimizes the weights of each feature extractor to maximize classification accuracy.

        Args:
            subset_size (int): Number of images to include in the subset.

        Returns:
            dict: Optimized weights for each feature extractor.
        """
        list_of_extractors = ["pose_estimator", "SIFT_descriptor_detector", "CNN"]
        weights = [1, 1, 1]    
        best_accuracy = 0
        best_weights = weights.copy()
        
        sys.path.append("feature_extraction")

        db_1, db_2, db_3 = self._read_dbs()

        image_files = []
        file_to_subfolder = {}
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
                    file_to_subfolder[os.path.basename(full_path)] = self.extract_subfolder(full_path)

        s = {2, 4, 6, 8, 10}
        assignments = {self.normalize_tuple((x, y, z)) for x in s for y in s for z in s}

        # Generate new weights combinations
        start = time.time()
        for test_weights in assignments:
            print(f'last iteration took {time.time() - start} seconds')
            start = time.time()
            test_weights = list(test_weights)
            print(Fore.RED + f'Testing weights: {test_weights}' + Fore.RESET)                        
            accuracy_count = 0 
            random_subset = random.sample(image_files, subset_size)         

            for first_path in random_subset:
                closest_image = ""
                best_score = float('-inf')

                pose_feature1 = db_2[db_2['Filename'] == os.path.basename(first_path)]['Feature'].iloc[0]
                sift_feature1 = db_1[db_1['Filename'] == os.path.basename(first_path)]['Feature'].iloc[0]
                cnn_feature1 = db_3[db_3['Filename'] == os.path.basename(first_path)]['Feature'].iloc[0]

                for i in db_1.index:
                    if db_1['Filename'][i] == os.path.basename(first_path):
                        continue
                    pose_feature2, sift_feature2, cnn_feature3 = db_2['Feature'][i], db_1['Feature'][i], db_3['Feature'][i]
                    score = self.calculate_score(os.path.basename(first_path), db_1['Filename'][i], test_weights, pose_feature1, sift_feature1, cnn_feature1, pose_feature2, sift_feature2, cnn_feature3)
                    if score > best_score:
                        best_score = score
                        closest_image = db_1['Filename'][i]
                if closest_image != "":
                    if file_to_subfolder[closest_image] == self.extract_subfolder(first_path):
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
    
    # User input for subset size
    subset_size = int(input("Enter the subset size: "))
    
    # Optimize weights
    optimized_weights = optimizer.optimize_weights(subset_size)
    print(optimized_weights)
