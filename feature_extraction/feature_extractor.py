import os
import pandas as pd
import numpy as np
from extractors.pose_estimator import pose_estimator
from extractors.sift_desc_detector import sift_desc_detector


class feature_extractor:
    '''
    A class reperesenting a feature extractor. The feature extractor has
    different extractor objects for specific extracting purposes.

    Attributes:
        - photo_path (Str): path of current photo
        - list_of_extractor (List[extractors]): List of all extractors

    Functions:
        - extract():                 Function for extracting each feature of an image
                                     utilizing all extractors.
        
        - compare():                 Function for comparing query image to a specific
                                     image given.

        - extractAndSave():          Function for extracting and saving DB Images

        - set_photo_path(new_path):  Function for setting the photo_path

        - get_photo_path():          Function for retrieving the photo_path
    '''
    # Class variables for storing photo path, feature_extractors
    # and all extractors in a list for future use.
    photo_path = None
    list_of_extractors = []

    pose_weight = None
    sift_weight = None
    cnn_weight = None


    # Initiate the feature extractor
    def __init__(self, photo_path="", sift_weight=0.2, pose_weight=0.3, cnn_weight=0.4):
        # Set the path of the photo and weights:
        self.photo_path = photo_path
        
        self.sift_weight = sift_weight
        self.pose_weight = pose_weight
        self.cnn_weight = cnn_weight

        # TODO: For every feature added to list of feature extractors
        self.list_of_extractors.append(sift_desc_detector(photo_path))
        self.list_of_extractors.append(pose_estimator(photo_path))



    def extract(self):
        '''
        Main function for calculating features. Only used for QUERY IMAGE. Also called
        by DB images during DB creation by extractAndSave().
        
        Returns:
            - (List[features]): A list of features
        '''
        # Extract features and return
        return [extractor.get_features() for extractor in self.list_of_extractors]



    def compare(self, image_to_compare_path):
        '''
        Main function for comparing to images to each other.

        Parameters:
            - image_to_compare_path (Str): String of the photo to compare agains query photo

        Returns:
            - Compare_output (Float):      Output after comparison between two photos
        '''
        list_of_compare_outputs = [extractor.compare(image_to_compare_path) for extractor in self.list_of_feature_extractors]

        # why not just compare_output = w1 * list_of_compare_outputs[0] + w2 * list_of_compare_outputs[1] + w3 * list_of_compare_outputs[2] ??
        
        # TODO: Weight outputs for example (go through each output and put a weight):
        for index, extractor, output in enumerate(zip(self.list_of_feature_extractors, list_of_compare_outputs)):
            if (extractor.get_name() == "SIFT_descriptor_detector"):
                list_of_compare_outputs[index] = self.sift_weight * output #weighted output
            elif (extractor.get_name() == "Pose_estimator"):
                list_of_compare_outputs[index] = self.pose_weight * output #weighted output
            else:
                list_of_compare_outputs[index] = self.cnn_weight * output #weighted output

        compare_output = sum(list_of_compare_outputs)

        return compare_output



    def extractAndSave(self):
        '''
        Main function used for extracting the features of DB Images and saving
        them to the corresponding DB.
        '''
        # Extract features and save to DB:
        list_of_features = self.extract()
        for i, extractor in enumerate(self.list_of_extractors):
            extractor_name = extractor.get_name()
            curr_features = list_of_features[i]

            # Check that we arent trunctuating the np.array:
            if i == 0: curr_features = np.array2string(curr_features.flatten(), threshold=np.inf, max_line_width=np.inf)

            # Creating a pandas dataframe
            curr_photo_path = os.path.join("archive", os.path.dirname(self.photo_path).split(os.path.sep)[-1], os.path.basename(self.photo_path))
            df = pd.DataFrame({'features': [curr_features],'photo_path': curr_photo_path})
            save_path = f"feature_DB/{extractor_name}_features.csv"

            # Check if the CSV file already exists, and append:
            file_exists = os.path.isfile(save_path)
            mode = "a" if file_exists else "w"  
            header = False if file_exists else True
            df.to_csv(save_path, sep="|", mode=mode, header=header, index=False)



    '''Setter function for setting photo path'''
    def set_new_photo(self, new_path):
        # Setting the new path and using list comp. to set extractors
        self.photo_path = new_path
        _ = [ex.set_new_photo(new_path) for ex in self.list_of_extractors]


    '''Getter function for getting photo path'''
    def get_photo_path(self):
        return self.photo_path