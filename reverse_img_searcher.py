from feature_extraction.feature_extractor import feature_extractor
import numpy as np

class reverse_img_searcher:
    '''
    A class representing a reverse image searcher. This reverse
    image searcher searches the picture uploaded to the repository
    DB and retrieves the best result.

    Attributes:
        - photo_path (Str): String of the current photo attached
    
    Functions:
        - ...():
    '''
    photo_path = None
    photo_feature_extractor = None

    def __init__(self, photo_path="", sift_w=0.2, pose_w=0.3, cnn_w=0.4):
        self.photo_path = photo_path

        # Initiate feature extractor and extract
        self.photo_feature_extractor = feature_extractor(photo_path,sift_weight=sift_w,pose_weight=pose_w,cnn_weight=cnn_w)
        self.photo_feature_extractor.extract()


    def search(self):
        comparisons = []

        # Read DB:
        db_1, db_2 = self.read_dbs()
        for i, (db_1_line, db_2_line) in enumerate(zip(db_1, db_2)):
            if i == 0:
                continue
            if i == 1:
                #db_1_line, db_2_line = db_1_line.split(","),  db_2_line.split(",")
                print(db_2_line)

        return 0
    

    def read_dbs(self):
        with open("./feature_db/SIFT_descriptor_detector_features.csv", "r") as db_1:
            db_1 = db_1.readlines()
        with open("./feature_db/Pose_estimator_features.csv", "r") as db_2:
            db_2 = db_2.readlines()
        return db_1, db_2


    def from_DB_to_array(self):
        return 0