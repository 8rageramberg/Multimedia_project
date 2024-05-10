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
    sift_nr_descriptors = None

    def __init__(self, photo_path="", sift_nr_descriptors=50, sift_w=0.2, pose_w=0.3, cnn_w=0.4):
        self.photo_path = photo_path
        self.sift_nr_descriptors = sift_nr_descriptors

        # Initiate feature extractor and extract
        self.photo_feature_extractor = feature_extractor(
            photo_path, sift_nr_descriptors=sift_nr_descriptors,
            sift_weight=sift_w, pose_weight=pose_w, cnn_weight=cnn_w)
        self.photo_feature_extractor.extract()


    def search(self):
        comparisons = []

        # Read DB:
        db_1, db_2 = self.read_dbs()
        for i, (db_1_line, db_2_line) in enumerate(zip(db_1, db_2)):
            if i == 0:
                continue
            if i == 1:
                # Stripping and splitting the DB lines with regards to "|" seperator
                db_1_line, db_2_line = db_1_line.strip().split("|"), db_2_line.strip().split("|")
                
                # Use eval function to parse from string to python object:
                sift_features = eval("np.array(" + db_1_line[0] + ", dtype=np.float32)").reshape(self.sift_nr_descriptors, 128)
                pose_features = eval("np.array(" + db_2_line[0] + ")").reshape(33, 4)

                # TODO: FIX COMPARISON
                comparisons.append(self.photo_feature_extractor.compare([sift_features, pose_features]))
                print(comparisons)
                

                

        return 0


    def read_dbs(self):
        with open("./feature_db/SIFT_descriptor_detector_features.csv", "r") as db_1:
            db_1 = db_1.readlines()
        with open("./feature_db/Pose_estimator_features.csv", "r") as db_2:
            db_2 = db_2.readlines()
        return db_1, db_2


    def from_DB_to_array(self):
        return 0