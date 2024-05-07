import os

# RUN TO CREATE DB
def main():
    user_input = input("DO YOU WANT TO INITIATE DB CREATION? (Type yes to initiate / q to quit): ")
    if user_input.lower() == "yes": 
        # retrieving the directory to use for testing
        directory = os.path.dirname((os.path.abspath(__file__)))
        directory = os.path.join(directory, "archive")

        # Iterating through images and making a list of image paths:
        images_to_use = []
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.basename(os.path.dirname(os.path.join(root, file))) == "Test_plank":
                    continue
                if os.path.basename(os.path.dirname(os.path.join(root, file))) == "Test_pullup":
                    continue
                images_to_use.append(os.path.join(root, file))

        # TODO: INITIATE DB CREATION

    elif user_input.lower() == "q":
        print("Goodbye!")

    else:
        print("I did not understand")
        main()


if __name__ == "__main__":
    main()
