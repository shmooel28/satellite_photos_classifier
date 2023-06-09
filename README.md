# Classify countries from satellite images using deep learning
        bar nahmias 312429327. shmuel ben david 208916437

![image](https://github.com/shmooel28/satellite_photos_classifier/assets/92825016/0587e73a-0e5a-4720-a5ed-df68f342d012)

https://drive.google.com/file/d/1CY0F_YkJ9GUzbAIzr92nVSuRsDeA-GUO/view?usp=drive_link


## Problem Description
Given a satellite video, we would like to create an algorithm that can decipher for us what geographic area we are seein


## Description

Satellite photo classification involves analyzing images captured by satellites orbiting the Earth and identifying the geographical locations depicted in the images. This project aims to leverage machine learning techniques to automate the classification process and accurately determine where the photos were taken.


## dataset:

NASA LIVE - day\night \clouds\ different angles (May-June 2023)  🌎 Earth From Space Live Stream : Live Views from the ISS - YouTube
Google Eath - day https://earth.google.com/web/
Google Eath Night - night https://earth.google.com/web/data=CiQSIhIgMGY3ZTJkYzdlOGExMTFlNjk5MGQ2ZjgxOGQ2OWE2ZTc
Govmap - Israel day 2022 https://www.govmap.gov.il/?c=204000,595000&z=0


To create a dataset, we sampled for several days video clips of the Mediterranean basin area, weather conditions, lighting, clouds and different angles.
We created a script in Python that accepts a video clip and extracts snapshots from it every X 
(default :  5 )  frames.
PyTorch's torchvision.transforms performed augmentation on the existing dataset.


![image](https://github.com/shmooel28/satellite_photos_classifier/assets/92825016/488b467f-c82b-4da2-988b-9aaa7e039e05)


## Description of the development plan of the project:


Description of the development plan of the project:

Review of the Literature

Creating a data set of satellite images of countries in the Mediterranean basin area, under varying lighting conditions, shooting angles and clouds

In our model we will work with a CNN type network, Convolutional Neural Network, we will be based on PyTorch, an open source machine learning library.

  	Loop over all frames in the video file
For each frame, pass the frame through the CNN
Obtain the predictions from the CNN
Maintain a list of the last K predictions
Compute the average of the last K predictions and choose the label with the largest corresponding probability
Label the frame and write the output frame to disk

Model training

Model evaluation

Detecting the speed of the camera and adding the speed in kilometers per hour to the video

![image](https://github.com/shmooel28/satellite_photos_classifier/assets/92825016/7ed3a8de-f7d1-4e92-807d-3086dfc22600)


## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

   ```shell
   $ git clone https://github.com/shmooel28/satellite_photo_classifier.git

## Usage

To classify satellite photos and determine their locations, follow these steps:

1. Create your dataset from videos. Run the extract_frames.py file to extract frames from the video and save them as photos. Use the following command:

(example) python extract_frames.py source_path output_path --frame_interval 3

- 'source_path' is the path to the video file.

- 'output_path' is the directory where the extracted photos will be saved.

- 'frame_interval' (optional) specifies the number of frames to skip between saved photos. The default value is 5.

2. In the train file you need to enter your main dir where the dataset is, than train the classification model using the preprocessed data:

$ python train.py

3. Test your model

$ python test.py

