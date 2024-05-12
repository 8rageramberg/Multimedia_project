import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "feature_extraction"))
from feature_extractor import feature_extractor
import time as t
import pandas as pd


# RUN TO CREATE DB
def main():
    # Initiate with user input:
    user_input = input("DO YOU WANT TO INITIATE DB CREATION? (Type yes to initiate / q to quit): ")
    if user_input.lower() == "yes":

        print("#### INITIATING: ####") 
        db_start_time = t.time()

        # RETRIEVING THE DIRECTORY TO USE:
        directory = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
        directory = os.path.join(directory, "archive")
        feature_extractor_db = feature_extractor()

        # INITIATE DATAFRAMES:
        list_of_feature_names = feature_extractor_db.get_names()
        list_of_dataframes = [pd.DataFrame(columns=['Feature', 'Filename']),
                              pd.DataFrame(columns=['Feature', 'Filename']),
                              pd.DataFrame(columns=['Feature', 'Filename'])]
        list_of_dataframes[0]['Feature'] = list_of_dataframes[0]['Feature'].astype(object)
        list_of_dataframes[1]['Feature'] = list_of_dataframes[1]['Feature'].astype(object)
        list_of_dataframes[2]['Feature'] = list_of_dataframes[2]['Feature'].astype(object)

        # START FILE WALKING
        for root, _, files in os.walk(directory):
            for counter, file in enumerate(files):
                absolute_file_path = os.path.join(root, file)
                parent_folder = os.path.basename(os.path.dirname(os.path.join(root, file)))
                if counter == 0: 
                    feature_extractor_db.set_new_photo(absolute_file_path)
                    if os.path.isfile("feature_DB/Pose_estimator_features.pkl"):
                        print("Folder DONE!\n")

                # IGNORE SPECIFIC FOLDERS:
                if parent_folder == "Test_plank":
                    continue
                if parent_folder == "Test_pullup":
                    continue
                if "DS_Store" in absolute_file_path:
                    continue

                # UPDATE TO CURR IMG, EXTRACT AND APPEND TO DATAFRAME:
                feature_extractor_db.set_new_photo(absolute_file_path)
                features = feature_extractor_db.extract()
                for feature, dataframe in zip(features, list_of_dataframes):
                    dataframe.loc[len(dataframe)] = [feature, os.path.basename(absolute_file_path)]

                if counter % 100 == 0:
                    db_curr_time = t.time()
                    print(f"Picture {counter} in folder '{parent_folder}' successfully saved to DB.\nTime used: {db_curr_time-db_start_time:.2f} seconds\n")
        
        # SAVE THE FULLY MADE DATAFRAMES AS .PKL FILES:
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(absolute_file_path))), "feature_db")
        _ = [dataframe.to_pickle(os.path.join(save_path, f_name + "_features.pkl")) for f_name, dataframe in zip(list_of_feature_names, list_of_dataframes)]

    elif user_input.lower() == "q":
        print("Goodbye!")

    else:
        print("I did not understand")
        main()





if __name__ == "__main__":
    main()