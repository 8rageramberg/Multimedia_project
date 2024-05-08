import os
import sys
sys.path.append("feature_extraction")
from feature_extraction.feature_extractor import feature_extractor

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# RUN TO CREATE DB
def main():
    counter = 0
    user_input = input("DO YOU WANT TO INITIATE DB CREATION? (Type yes to initiate / q to quit): ")
    if user_input.lower() == "yes": 
        # Retrieving the directory to use for testing
        directory = os.path.dirname((os.path.abspath(__file__)))
        directory = os.path.join(directory, "archive")

        for root, _, files in os.walk(directory):
            for file in files:
                if counter % 100 == 0:
                    print("1000 done fuckface")

                absolute_file_path = os.path.join(root, file)
                parent_folder = os.path.basename(os.path.dirname(os.path.join(root, file)))

                # Ignore spesific folders and files
                if parent_folder == "Test_plank":
                    continue
                if parent_folder == "Test_pullup":
                    continue
                if "DS_Store" in absolute_file_path:
                    continue
                
                feature_extractor_db = feature_extractor(absolute_file_path)
                #feature_extractor_db.set_photo_path()

                # Save and extract:
                feature_extractor_db.extractAndSave()
                counter += 1

    elif user_input.lower() == "q":
        print("Goodbye!")

    else:
        print("I did not understand")
        main()


if __name__ == "__main__":
    main()
