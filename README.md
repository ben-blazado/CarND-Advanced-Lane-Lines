# Advanced Lane Finding (ALF)

This is a Udacity Self-Driving Car NanoDegree project submission that uses the following to detect road lanes:
- Lens calibration
- Distortion correction
- Edge detection
- Colorspace Transformation 
- Perspective transformation
- Pixel dection and Line Fitting 

![](output_images/wup_challenge_video.png)

## Installation
Clone or fork this repository.

## Usage
Intended user is the Udacity evaluator for this project. It is intended to be used in a Jupyter Notebook. Open `alf.ipynb` for examples of usage.

## Files
### Project Files
- `writeup.md`: writeup of project for Udacity evaluator; includes images of pipeline stages
- `output_images/project_video.mp4`: project video for submission
- `alf_*.py` files: python code for project
  - `alf_con.py`: main controller for sequencing pipeline stages
  - `alf_cam.py`: camera for calibration and distortion correction
  - `alf_enh.py`: enhancer for edge detection and HSV color transformation
  - `alf_war.py`: warper for perpective transformation
  - `alf_llg.py`: lane finders, sliding and linear window search areas, and line to find and annotate lanes
  - `alf_hud.py`: heads up display (simulated) for composing final image with lane area
  - `alf_utils.py`: logging and demonstration functions
 
### Other files 
- `alf.ipynb`: jupyter notebook on project usage and demonstrations of important functions
- `output_images` folder:
  - `wup_*.png`: images used in writeup
  - `project_video.mp4`: project video for submission 
  - `challenge_video.pm4`: pipeline worked on challenge video too
  - `harder_challenge_video.mp4`: pipeline failed on this harder challenge video
  - `out_*_video_.mp4`: output of videos processed in `alf.ipynb` examples; prevents overwriting project videos for submission
  - `straight_lines1-undist.jpg`: image used to find and plot srcpoints for perspective transformation
- `sandbox.ipynb`: jupyter notebook for testing code
- `sketch.drawio`: UMLish sketch of component collaborations; requires [diagrams.net](https://www.diagrams.net/) to view; does not reflect current state of components
- `adv_lane_fine.log`: debug log of project