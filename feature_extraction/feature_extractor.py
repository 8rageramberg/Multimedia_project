from .extractors.pose_estimator import pose_estimator
from .extractors.sift_desc_detector import sift_desc_detector


class feature_extractor:
    '''
    A class reperesenting a feature extractor. The feature extractor has
    different extractor objects for specific extracting purposes.

    Attributes:
        - photo_path (Str): path of current photo
        - list_of_extractor (List[extractors]): List of all extractors

    Functions:
        - extract():                        Function for extracting each feature of an image
                                            utilizing all extractors.
        
        - compare(image_to_compare_path):   Function for comparing query image to image given

        - extractAndSave():                 Function for extracting and saving DB Images

        - set_photo_path(new_path):         Function for setting the photo_path

        - get_photo_path():                 Function for retrieving the photo_path
    '''
    # Class variables for storing photo path, feature_extractors
    # and all extractors in a list for future use.
    photo_path = None
    list_of_extractors = []


    # Initiate the feature extractor
    def __init__(self, photo_path=""):
        # Set the path of the photo
        self.photo_path = photo_path

        # TODO: For every feature added to list of feature extractors
        self.list_of_feature_extractors.append(pose_estimator(photo_path))
        self.list_of_feature_extractors.append(sift_desc_detector(photo_path))



    def extract(self):
        '''
        Main function for calculating features. Only used for QUERY IMAGE. Also called
        by DB images during DB creation by extractAndSave().
        
        Returns:
            - (List[features]): A list of features
        '''
        # Extract features and return
        return [extractor.get_features() for extractor in self.list_of_feature_extractors]



    def compare(self, image_to_compare_path):
        '''
        Main function for comparing to images to each other.

        Parameters:
            - image_to_compare_path (Str): String of the photo to compare agains query photo

        Returns:
            - Compare_output (Float):      Output after comparison between two photos
        '''
        list_of_compare_outputs = [extractor.compare(image_to_compare_path) for extractor in self.list_of_feature_extractors]
        
        # TODO: Weight outputs for example (go through each output and put a weight):
        for index, extractor, output in enumerate(zip(self.list_of_feature_extractors, list_of_compare_outputs)):
            if (extractor.get_name() == "SIFT descriptor detector"):
                list_of_compare_outputs[index] = 0.2 * output #weighted output
            elif (extractor.get_name() == "Other extractor"):
                list_of_compare_outputs[index] = 0.3 * output #weighted output
            else:
                list_of_compare_outputs[index] = 0.4 * output #weighted output

        compare_output = sum(list_of_compare_outputs)

        return compare_output



    def extractAndSave(self):
        '''
        Main function used for extracting the features of DB Images and saving
        them to the corresponding DB.
        '''
        features_to_save = self.extract()
        # TODO: Save features to DB!




    '''Setter function for setting photo path'''
    def set_photo_path(self, new_path):
        self.photo_path = new_path


    '''Getter function for getting photo path'''
    def get_photo_path(self):
        return self.photo_path