import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class extractor_2:
    name = "Extractor 2"
    photo_path = None

    def __init__(self, photo_path=""):
        self.photo_path = photo_path
        # Load a pre-trained ResNet-50 model
        self.model = models.resnet50(pretrained=True)
        # Remove the last fully connected layer (classification layer)
        self.model.fc = torch.nn.Identity()
        # Set the model to evaluation mode
        self.model.eval()

    def get_features(self):
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load and preprocess the image
        image = Image.open(self.photo_path)
        image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

        # Extract features using the ResNet-50 model
        with torch.no_grad():
            features = self.model(image_tensor)

        # Convert features to a NumPy array (if needed)
        features = features.numpy()

        return features

    def get_name(self):
        return self.name