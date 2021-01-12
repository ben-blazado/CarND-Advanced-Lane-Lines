# Advanced Lane Finding (ALF)
This is a Udacity Self-Driving Car NanoDegree project submission that uses the following to detect road lanes:
- Lens calibration
- Distortion correction
- Edge detection
- Colorspace Transformation 
- Perspective transformation
- Pixel dection and Line Fitting 

## Installation
Clone or fork this repository.

## Usage
This project is intended to be used in a Jupyter Notebook. Open alf.ipynb for examples of usage.

## Important Files
- writeup.md: writeup of project for Udacity evaluator; includes images of piline stages
- output_images/project_video_lane_area_submit.mp4: project video for submission
- alf\_\*.py files: python code for project
  - alf_con.py: main controller for sequencing pipeline stages
  - alf_cam.py: camera for calibration and distortion correction
  - alf_war.py: warper for perpective transformation
  - alf_llg.py: lane finders, sliding and linear window search areas, and line to find lane lines
  - alf_hud.py: heads up display (simulated) for composing final image with lare area
  - alf_utils.py: logging and demonstration functions
 
### Other files 
- output_images folder:
 - wup\_\*.png: images used in writeup
 - project_video_lane_area_submit.mp4: project video for submission 
- alf.ipynd: jupyter notebook for example of using project
- sandbox.ipynb: jupyter notebook for testing code
- sketch.drawio: UML sketch of components; requires [diagrams.net](https://www.diagrams.net/) to view; does not reflect current state of components
- adv_lane_fine.log: debug log of project
    

