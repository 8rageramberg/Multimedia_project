import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "feature_extraction"))
from feature_extractor import feature_extractor
import time as t



# RUN TO CREATE DB
def main():
    # Initiate with user input:
    user_input = input("DO YOU WANT TO INITIATE DB CREATION? (Type yes to initiate / q to quit): ")
    if user_input.lower() == "yes":
        print("#### INITIATING: ####") 
        db_start_time = t.time()
        # Retrieving the directory to use:
        directory = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
        directory = os.path.join(directory, "archive")
        feature_extractor_db = feature_extractor()

        for root, _, files in os.walk(directory):
            for counter, file in enumerate(files):
                absolute_file_path = os.path.join(root, file)
                parent_folder = os.path.basename(os.path.dirname(os.path.join(root, file)))
                if counter == 0: 
                    feature_extractor_db.set_new_photo(absolute_file_path)
                    if os.path.isfile("feature_DB/Pose_estimator_features.csv"):
                        print("Folder DONE!\n")

                # Ignore specific folders and files:
                if parent_folder == "Test_plank":
                    continue
                if parent_folder == "Test_pullup":
                    continue
                if "DS_Store" in absolute_file_path:
                    continue

                # Update, save and extract:
                feature_extractor_db.set_new_photo(absolute_file_path)
                feature_extractor_db.extractAndSave()
                if counter % 100 == 0:
                    db_curr_time = t.time()
                    print(f"Picture {counter} in folder '{parent_folder}' successfully saved to DB.\nTime used: {db_curr_time-db_start_time:.2f} seconds\n")

    elif user_input.lower() == "q":
        print("Goodbye!")

    else:
        print("I did not understand")
        main()


if __name__ == "__main__":
    main()
