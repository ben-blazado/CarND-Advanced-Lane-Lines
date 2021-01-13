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
- `alf\_\*.py` files: python code for project
  - `alf_con.py`: main controller for sequencing pipeline stages
  - `alf_cam.py`: camera for calibration and distortion correction
  - `alf_enh.py`: enhancer for edge detection and HSV color transformation
  - `alf_war.py`: warper for perpective transformation
  - `alf_llg.py`: lane finders, sliding and linear window search areas, and line to find and annotate lanes
  - `alf_hud.py`: heads up display (simulated) for composing final image with lare area
  - `alf_utils.py`: logging and demonstration functions
 
### Other files 
- `alf.ipynb`: jupyter notebook for example of using project and demonstrations of important functions
- output_images folder:
  - `wup\_\*.png`: images used in writeup
  - `project_video.mp4`: project video for submission 
  - `challenge_video.pm4`: pipeline worked on challege video too
  - `harder_challenge_video.mp4`: pipeline failed on this harder challenge video
  - `out\_\*\_video_.mp4` : videos processed in the examples in alf.ipynb
  - `straight_lines1-undist.jpg`: image used to find src points in perspective transformation
- `sandbox.ipynb`: jupyter notebook for testing code
- `sketch.drawio`: UML sketch of components; requires [diagrams.net](https://www.diagrams.net/) to view; does not reflect current state of components
- `adv_lane_fine.log`: debug log of project