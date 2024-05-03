import cv2
import numpy as np

'''
Class representing an edge detector.

Attributes:
    - photo_url (Str): URL of current photo
'''
class edge_detector:
    name = "Edge Detector"
    photo_url = None

    def __init__(self, photo_url=""):
        self.photo_url = photo_url
        
    '''
    Main func:
    '''
    def get_features(self):
        
        edges = self.get_edges(self, img_size = (1000, 1000))
        features = {
            'edge_hist': cv2.calcHist([edges], [0], None, [256], [0, 256]),
            'edge_density': np.sum(edges) / (edges.shape[0] * edges.shape[1])
        }
        # Save the edges as an image
        print("Shape of edges array:", edges.shape)
        return features
         
    '''Getter function for getting extractor name'''
    def get_name(self):
        return self.name
    

    def get_edges(self, display=False, img_size = (1000, 1000)):
        # Read the image
        image = cv2.imread(self.photo_url, 0)  # Read as grayscale
        image_scaled = cv2.resize(image, img_size)

        # Apply Canny edge detection
        threshold_1 = 50  # Lower threshold
        threshold_2 = 150  # Upper threshold

        edges = cv2.Canny(image_scaled, threshold_1, threshold_2)
        print(cv2.calcHist([edges], [0], None, [256], [0, 256]))
        if display:
            cv2.imshow('normal', image)
            cv2.waitKey(0)  # Press a key to remove the window
            cv2.destroyAllWindows()

            cv2.imshow('resized', image_scaled) 
            cv2.waitKey(0)  # Press a key to remove the window
            cv2.destroyAllWindows()

            cv2.imshow('Edges', edges)
            cv2.waitKey(0)  # Press a key to remove the window
            cv2.destroyAllWindows()

        return edges

# Uncomment to see canny edge detection
# test_1 = 'archive/A_test_set/Test_plank/plank_g3_copy.jpg'
# detector = edge_detector(test_1)
# features = detector.get_features()
  