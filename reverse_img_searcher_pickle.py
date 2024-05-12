from feature_extraction.feature_extractor import feature_extractor
import numpy as np
import time as t
import os
import pandas as pd

class reverse_img_searcher_pickle:
    '''
    A class representing a reverse image searcher. This reverse
    image searcher searches the picture uploaded to the repository
    DB and retrieves the best result.

    Attributes:
        - photo_path (Str):          String of the current photo attached
        - sift_nr_descriptors (Int): Number of sift descriptors to use, def = 50
        - sift_w (Int):              Weight of sift output when comparing, def = 0.2
        - pose_w (Int):              Weight of pose estimation output when comparing, def = 0.3
        - cnn_w (Int):               Weight of CNN output when comparing, def = 0.4
    
    Functions:
        - search():         Initiate reverse image search with query image.
        - _compare_to_db(): Private function called to compare query image to db. 
        - _read_dbs():      Private function called by search to retrieve feature dbs.
    '''
    photo_path = None
    photo_feature_extractor = None
    sift_nr_descriptors = None

    def __init__(self, photo_path="", sift_nr_descriptors=50, sift_w=0.2, pose_w=0.3, cnn_w=0.4):
        self.photo_path = photo_path
        self.sift_nr_descriptors = sift_nr_descriptors

        # Initiate feature extractor and extract
        self.photo_feature_extractor = feature_extractor(
            photo_path, sift_nr_descriptors=sift_nr_descriptors,
            sift_weight=sift_w, pose_weight=pose_w, cnn_weight=cnn_w)
        self.photo_feature_extractor.extract()


    def search(self, nr_of_pics=5):
        '''
        This function starts the reverse image search utilizing the 
        feature pkl databases and a feature extractor object to utilize
        the features of the current query image

        Parameters:
            - nr_of_pics (Int): Number of pics to retrieve, def = 5

        Returns:
            - comparisons_as_np_arr_sort[-nr_of_pics] (Slice of np.array): sorted np array with the nr_of_pics most
                                                                           alike photos.
        '''
        # PRINT OUT ALGO TIME:
        print(f"\n####### STARTING SEARCH #######")
        print(f"Picture used: {self.photo_path}\n\nWeights:\nSift weight: {self.photo_feature_extractor.sift_weight}\nPose weight: {self.photo_feature_extractor.pose_weight}\nCNN weight: {self.photo_feature_extractor.cnn_weight}\n\n########### RESULTS: ##########")
        algo_start_time = t.time()

        # Read the image DB and compare:
        comparisons = []
        db_1, db_2, db_3 = self._read_dbs()
        self._compare_to_db(db_1, db_2, db_3, comparisons)
        
        # Transform comparison array to np array and argsort:
        comparisons_as_np_arr = np.array(comparisons)
        comparisons_as_np_arr_sort = comparisons_as_np_arr[np.argsort(comparisons_as_np_arr[:, 0])]
        
        # Print algo time to terminal and return the nr_of_pics photos with heighest comparison:
        algo_end_time = t.time()
        print(f"The reverse image search took: {algo_end_time-algo_start_time:.2f} seconds") 
        return comparisons_as_np_arr_sort[-nr_of_pics:] if nr_of_pics <= len(comparisons_as_np_arr_sort) else comparisons_as_np_arr_sort



    def _compare_to_db(self, db_1, db_2, db_3, comparisons):
        '''
        Private function for comparing the query image towards the feature db

        Parameters:
            - db_1 (pd.Dataframe):        First feature db as a Dataframe
            - db_2 (pd.Dataframe):        Second feature db as a Dataframe
            - db_3 (pd.Dataframe):        Third feature db as a Dataframe
            - comparisons (List[tuples]): List of tuples to append comparison and pic name to
        '''
        for i in db_1.index:
            comparisons.append((self.photo_feature_extractor.compare([db_1['Feature'][i], db_2['Feature'][i], db_3['Feature'][i]]), db_1['Filename'][i]))
                

    def _read_dbs(self):
        ''' Private function for reading the DB'''
        pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_db")

        db_1 = pd.read_pickle(os.path.join(pkl_path, "SIFT_descriptor_detector_features.pkl"))
        db_2 = pd.read_pickle(os.path.join(pkl_path, "Pose_estimator_features.pkl"))
        db_3 = pd.read_pickle(os.path.join(pkl_path, "CNN_features.pkl"))
        
        return db_1, db_2, db_3