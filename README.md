# AudioVisualizer# AudioVisualizer

This script generates a video with an audio visualizer from an audio file and an optional background image.

## Features

-   Generates waveform or spectrum visualizations.
-   Customizable colors, dimensions, FPS, and other visual parameters.

## Prerequisites

Make sure you have Python installed. You will also need FFmpeg for `moviepy` to function correctly.

## Installation

1.  Clone the repository or download the `visualizer.py` script.
2.  Install the required Python packages:
    ```sh
    pip install librosa numpy opencv-python tqdm moviepy==1.0.3
    ```

## Configuration

The script uses a `CONFIG` dictionary within [visualizer.py](d:\Projects\AudioVisualizer\visualizer.py) to set various parameters:

-   `audio_file`: Path to the input audio file (e.g., "input.mp3").
-   `output_video`: Path for the generated video file (e.g., "output.mp4").
-   `background_image_path`: Path to an image to use as the background (e.g., "background.jpg"). If not found, a solid color background is used.
-   `fps`: Frames per second for the output video.
-   `video_width`, `video_height`: Dimensions of the output video.
-   `visualizer`: Contains settings for the visualizer type (`waveform` or `spectrum`) and its specific parameters.
    -   `waveform`: Settings for line color, thickness, height, style, and smoothing.
    -   `spectrum`: Settings for number of bins, colormap, bar width, smoothing, frequency range, scaling, and gravity effect.
-   `ffmpeg_params`: Parameters for FFmpeg, including codecs, preset, threads, and custom arguments like CRF and audio bitrate.

Modify the `CONFIG` dictionary in [visualizer.py](d:\Projects\AudioVisualizer\visualizer.py) to suit your needs.

## Usage

1.  Ensure your input audio file (e.g., `input.mp3`) and optional background image (e.g., `background.jpg`) are in the same directory as the script, or update the paths in the `CONFIG` section of [visualizer.py](d:\Projects\AudioVisualizer\visualizer.py).
2.  Run the script from your terminal:
    ```sh
    python visualizer.py
    ```
3.  The output video (e.g., `output.mp4`) will be generated in the same directory.

## Files in the Project

-   `visualizer.py`: The main Python script for generating the audio visualization.
-   `input.mp3`: (Example) Your input audio file.
-   `background.jpg`: (Example) Your optional background image.
-   `output.mp4`: (Example) The generated video output.
-   `README.md`: This file.