# Satellite Photo Classifier

This project focuses on classifying satellite photos of the Earth taken from space and determining their locations on the planet.

## Description

Satellite photo classification involves analyzing images captured by satellites orbiting the Earth and identifying the geographical locations depicted in the images. This project aims to leverage machine learning techniques to automate the classification process and accurately determine where the photos were taken.

## Features

- Image classification: Utilize deep learning models to classify satellite photos based on their contents, such as landforms, urban areas, bodies of water, etc.
- Location identification: Determine the geographical coordinates (latitude and longitude) corresponding to each classified satellite photo.
- Interactive visualization: Provide an interactive interface to visualize the satellite photos along with their predicted locations on a map.

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

