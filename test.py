import torch
from torchvision.io import read_image
import sys
import torchvision.transforms.functional as TF

from Classifier import Classifier


# Load the saved model
model = Classifier()
model.load_state_dict(torch.load('model.pth'))
model.eval()
state_labels = {
    'Israel':0,
    'Lebanon':1,
    'Cyprus':2,
    'Greece':3,
    'Egypt':4,
    'Italy':5,
    'Spain':6,
    'Jordan':7
}
# Define the class labels
class_labels = ['Cyprus', 'Egypt', 'Greece', 'Israel', 'Italy', 'Jordan', 'None','Spain','Turkey']

# Get the image path from the command line argument
image_path = sys.argv[1]

# Read and preprocess the image
image = read_image(image_path)
image = image.to(torch.float32)
resized_tensor = TF.resize(image, (32, 32))
normalized_tensor = TF.normalize(resized_tensor, mean=0.5, std=0.5)

# Perform the inference
with torch.no_grad():
    output = model(normalized_tensor.unsqueeze(0))
    print(output)
    _, predicted = torch.max(output.data, 1)
    predicted_label = class_labels[predicted.item()]

# Print the predicted rice type
print(predicted_label)