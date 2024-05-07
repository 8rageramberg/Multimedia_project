import os
from feature_extraction import feature_extractor

# RUN TO CREATE DB
def main():
    user_input = input("DO YOU WANT TO INITIATE DB CREATION? (Type yes to initiate / q to quit): ")
    if user_input.lower() == "yes": 
        # retrieving the directory to use for testing
        directory = os.path.dirname((os.path.abspath(__file__)))
        directory = os.path.join(directory, "archive")

        for root, _, files in os.walk(directory):
            for file in files:
                absolute_file_path = os.path.join(root, file)
                parent_folder = os.path.basename(os.path.dirname(os.path.join(root, file)))

                if parent_folder == "Test_plank":
                    continue
                if parent_folder == "Test_pullup":
                    continue

                # Initiate the feature extractor:
                feature_extractor_db = feature_extractor()
                feature_extractor_db.set_photo_path(absolute_file_path)

                # Save and extract:
                feature_extractor_db.extractAndSave()
                #feature_extractor_db.destroy() destory the curr object for memory issues


    elif user_input.lower() == "q":
        print("Goodbye!")

    else:
        print("I did not understand")
        main()


if __name__ == "__main__":
    main()
