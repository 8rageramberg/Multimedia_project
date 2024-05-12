from feature_extraction.feature_extractor import feature_extractor
import numpy as np
import time as t
import os

class reverse_img_searcher:
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
        - search():    Initiate reverse image search with query image. 
        - _read_dbs(): Private function called by search to retrieve feature dbs.
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
        feature database and a feature extractor object to utilize
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
        Private function for comparing the query image
        towards the feature db

        Parameters:
            - db_1 (List[Str]):           First feature db as a list of strings
            - db_2 (List[Str]):           Second feature db as a list of strings
            - db_3 (List[Str]):           Second feature db as a list of strings
            - comparisons (List[tuples]): List of tuples to append comparison and pic name to
        '''
        for i, (db_1_line, db_2_line, db_3_line) in enumerate(zip(db_1, db_2, db_3)):
            # If the line is not a header, compare db features against
            # the query features.
            if i == 0:
                continue
            else:
                # Stripping and splitting the DB lines with regards to "|" seperator
                db_1_line, db_2_line, db_3_line = db_1_line.strip().split("|"), db_2_line.strip().split("|"), db_3_line.strip().split("|")
                
                # Use eval function to parse from string to python object:
                photo_name = db_1_line[1]
                try:
                    sift_features = eval("np.array(" + db_1_line[0] + ", dtype=np.float32)").reshape(self.sift_nr_descriptors, 128)
                    if db_2_line[0] != "None": pose_features = eval("np.array(" + db_2_line[0] + ")").reshape(33, 4)
                    else: pose_features = None
                    cnn_features = eval("np.array(" + db_3_line[0] + ")")
                except (ValueError):
                    print(f"Photo: {os.path.basename(db_1_line[1])}, gave ValueError, continuing!")
                    continue
                # Compare to DB:
                comparisons.append((self.photo_feature_extractor.compare([sift_features, pose_features, cnn_features]), photo_name))
                

    def _read_dbs(self):
        '''Read the feature DB and retrieve a list of each'''
        with open("./feature_db/SIFT_descriptor_detector_features.csv", "r") as db_1:
            db_1 = db_1.readlines()
        with open("./feature_db/Pose_estimator_features.csv", "r") as db_2:
            db_2 = db_2.readlines()
        with open("./feature_db/CNN_features.csv", "r") as db_3:
            db_3 = db_3.readlines()
        return db_1, db_2, db_3