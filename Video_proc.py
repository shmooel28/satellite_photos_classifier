import torch
import cv2
from torchvision.io import VideoReader, write_video
from torchvision.transforms.functional import normalize
import sys
import torchvision.transforms.functional as TF
from PIL import Image
from Classifier import Classifier

# Assuming you have the Classifier class and the path to the saved model

# Load the saved model
model = Classifier()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define the class labels
class_labels = ['Cyprus', 'Egypt', 'Greece', 'Israel', 'Italy', 'Jordan', 'None','Spain','Turkey']

# Define the number of previous predictions to consider
K = 10

# Get the video file path from the command line argument
video_path = sys.argv[1]

# Read the video using VideoReader
video = VideoReader(video_path)

frames = []
for frame in video:
    frames.append(frame['data'])
    break

# Define the output video path
output_path = 'output_video(2).mp4'

# Define the codec for output video
#fourcc = 'mp4v'

#output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc), video.get_metadata()["video"]["fps"], (frames[0].size()[1], frames[0].size()[2]))
#output_video = write_video(output_path, video["data"], video.get_metadata()["video"]["fps"], fourcc)
video2 = cv2.VideoCapture(video_path)
frame_width = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video2.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize the list of previous predictions
previous_predictions = []

# Iterate over each frame in the video
for frame in video:
    # Get the frame
    frame = frame['data'].float()
    ret, frame2 = video2.read()

    # Convert frame to PIL Image

    # Preprocess the frame
    #tensor = TF.to_tensor(frame)
    resized_tensor = TF.resize(frame, (32, 32))
    normalized_tensor = TF.normalize(resized_tensor, mean=0.5, std=0.5)

    # Perform the inference
    with torch.no_grad():
        output = model(normalized_tensor.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(probabilities.data, 1)
        predicted_label = class_labels[predicted.item()]

    # Print the predicted label
    print(predicted_label)

    # Add the current prediction to the list of previous predictions
    previous_predictions.append(probabilities.numpy())

    # Maintain the last K predictions
    if len(previous_predictions) > K:
        previous_predictions.pop(0)

    # Compute the average of the last K predictions
    average_predictions = sum(previous_predictions) / len(previous_predictions)

    # Choose the label with the highest average probability
    predicted_average = torch.argmax(torch.from_numpy(average_predictions))

    # Label the frame
    predicted_average_label = class_labels[predicted_average.item()]
    labeled_frame = cv2.putText(frame2, predicted_average_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the labeled frame to the output video
    output_video.write(labeled_frame)

# Release resources
video2.release()
output_video.release()
    #labeled_frame = cv2.putText(frame.numpy().astype('uint8'), predicted_average_label, (10, 30),
                              #  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the labeled frame to the output video
    #output_video.write(frame)

# Release resources
#video.close()
#output_video.release()


# Release resources
