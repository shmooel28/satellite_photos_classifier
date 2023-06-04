#python extract_frames.py source_path output_path --frame_interval 3
import random

import cv2
import argparse

def extract_frames(video_path, output_path, frame_interval):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    # Read the first frame
    success, frame = video.read()

    while success:
        # Save the frame to the output path
        random_num = random.randint(0,10000)
        output_frame_path = f"{output_path}/frame_{random_num}{frame_count}.jpg"
        cv2.imwrite(output_frame_path, frame)

        # Move to the next frame
        for _ in range(frame_interval):
            success, frame = video.read()
            frame_count += 1

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Extract frames from a video.')
    parser.add_argument('source_path', type=str, help='Path to the video file')
    parser.add_argument('output_path', type=str, help='Output path to save the frames')
    parser.add_argument('--frame_interval', type=int, default=5, help='Interval between frames')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Extract the arguments
    video_path = args.source_path
    output_path = args.output_path
    frame_interval = args.frame_interval

    extract_frames(video_path, output_path, frame_interval)
