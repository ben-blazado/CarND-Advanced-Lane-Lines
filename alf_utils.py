'''
Utility functions for use in jupyter notebook

Functions:
- demoChessboardCorners
- demoCameraCalibration
- demoEnhance
- demoWarpImage
- demoLaneSearch
- demoCompose
'''
import alf_cam
import alf_enh
import alf_war
import alf_llg
import alf_hud
import matplotlib.pyplot as plt
import cv2
import numpy as np
import logging, sys

# module wide logger
utils_logger = None 

def init_logging():
    '''
    Initializes logging for this project.
    '''
    
    global utils_logger
    
    #--- refs: code taken from: https://docs.python.org/3/howto/logging-cookbook.html
    #--- https://stackoverflow.com/a/9321890
    #--- https://stackoverflow.com/a/56366848   to clear any existing loggers
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logger = logging.getLogger()

    logging.basicConfig(
        level    = logging.DEBUG, 
        format   = '%(asctime)s | %(levelname)-8s | %(name)s: %(message)s',
        filename = 'adv_lane_find.log',
        filemode = 'w'
    )
        
    #--- console: INFO only, abbreviated format
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)-8s | %(name)s: %(message)s')
    console.setFormatter(console_formatter)
    logger.addHandler(console)
    
    utils_logger = logging.getLogger("alf_utils")
    utils_logger.info("Logging initialized.")
    
    return
    
def shutdown_logging():

    global utils_logger

    utils_logger.info("Logging shutdown.")
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    logging.shutdown()
    
    return
    

def createUndistStraight():
    '''
    Creates an undistorted image file of 'test_images/straight_lines1.jpg'.
    
    Notes:
    - Saves file to 'output_images\straight_lines1-undist.jpg'
    - Use output to define region for perspective transformation
    '''

    img = plt.imread("test_images/straight_lines1.jpg")
    cam = alf_cam.Camera()
    cam.calibrate()
    img_undist = cam.undistort(img)
    
    plt.figure(figsize=(17,6))
    ax1 = plt.subplot(121)
    ax1.imshow(img)
    ax1.set_title('Original Image with Barrel Distortion')
    ax2 = plt.subplot(122)
    ax2.imshow(img_undist)
    ax2.set_title('Corrected Image after Calibration')
    
    plt.imsave("output_images\straight_lines1-undist.jpg", img_undist)
    
    return


def demoChessboardCorners():
    '''
    Show image of chessboard corners annoted.
    
    Notes:
    - For use in Jupyter notebook.
    '''

    cal_set = alf_cam.ChessboardCameraCalibrationSet()
    cal_set.loadChessboardImages()
    cal_set.findChessboardCorners()
    cal_set.showCorners()
    
    return
    
    
def demoCameraCalibration():
    '''
    Shows before and after image of camera calibration
    
    Notes:
    - For use in Jupyter Notebook.
    '''

    img = plt.imread('camera_cal\calibration1.jpg')

    cam = alf_cam.Camera()
    cam.calibrate()
    img_undist = cam.undistort(img)
    
    plt.figure(figsize=(17,6))
    ax1 = plt.subplot(121)
    ax1.imshow(img)
    ax1.set_title('Original Image with Barrel Distortion')
    ax2 = plt.subplot(122)
    ax2.imshow(img_undist)
    ax2.set_title('Corrected Image after Calibration')

    return
    

def demoEnhance(filename):
    '''
    Shows before and after image of edge and color enhancement.
    '''

    img = plt.imread(filename)
    
    enh = alf_enh.Enhancer()
    enh.setParams(5, 76, 96, 30, 202)
    mask = enh.enhance(img)
    
    
    plt.figure(figsize=(17,6))
    
    ax1 = plt.subplot(121)
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image')
    
    ax2 = plt.subplot(122)
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Edge and Color Mask')    
    
    return    
    

def demoWarpImage(filename):
    '''
    Shows before and after image of image perspective transformation.
    '''

    img = plt.imread(filename)

    cam = alf_cam.Camera()
    cam.calibrate()
    img_undist = cam.undistort(img)
    
    war = alf_war.ImageWarper()
    war.calibrate()
    
    img_orig = np.copy(img_undist)
    img_orig = war.drawSrc(img_orig)
    
    img_warped = war.warpPerspective(img_undist)
    img_warped = war.drawDst(img_warped)
    
    plt.figure(figsize=(17,6))
    
    ax1 = plt.subplot(121)
    ax1.imshow(img_orig)
    ax1.set_title('Original Image')
    
    ax2 = plt.subplot(122)
    ax2.imshow(img_warped)
    ax2.set_title('Warped Imaged')
    
    return
        

def demoLaneSearch(filename):
    '''
    Shows sliding and linear windows search areas and lane areas.
    '''

    img = plt.imread(filename)

    cam = alf_cam.Camera()
    cam.calibrate()
    
    enh = alf_enh.Enhancer()
    enh.setParams(5, 76, 96, 30, 202)
    
    war = alf_war.ImageWarper()
    war.calibrate()
    
    alf = alf_llg.AdvancedLaneFinder()    
    alf.setParams(50, 3, 12, 0.9)
    
    img_undistorted      = cam.undistort(img)
    binary               = enh.enhance(img_undistorted)    
    binary_warped        = war.warpPerspective(binary)
    lane_area1, rad, off = alf.paintLaneArea(binary_warped, True)
    lane_area2, rad, off = alf.paintLaneArea(binary_warped, True)
    lane_area3, rad, off = alf.paintLaneArea(binary_warped)
    
    plt.figure(figsize=(40,20))
    
    ax1 = plt.subplot(311)
    ax1.imshow(lane_area1)
    ax1.set_title('Sliding Window')
    
    ax2 = plt.subplot(312)
    ax2.imshow(lane_area2)
    ax2.set_title('Linear Window')
    
    ax3 = plt.subplot(313)
    ax3.imshow(lane_area3)
    ax3.set_title('Lane Area')
    
    return
    
def demoCompose(filename):
    '''
    Shows original and final road image with lane areas annotated
    '''

    img = plt.imread(filename)

    cam = alf_cam.Camera()
    cam.calibrate()
    
    enh = alf_enh.Enhancer()
    enh.setParams(5, 76, 96, 30, 202)
    
    war = alf_war.ImageWarper()
    war.calibrate()
    
    alf = alf_llg.AdvancedLaneFinder()    
    alf.setParams(50, 3, 12, 0.9)
    
    hud = alf_hud.HUD()
    
    img_undistorted     = cam.undistort(img)
    binary              = enh.enhance(img_undistorted)    
    binary_warped       = war.warpPerspective(binary)
    lane_area, rad, off = alf.paintLaneArea(binary_warped)
    unwarped_lanes      = war.unwarpPerspective(lane_area)
    final_img           = hud.compose(img_undistorted, unwarped_lanes, rad, off)
    
    plt.figure(figsize=(17,6))
    ax1 = plt.subplot(121)
    ax1.imshow(img_undistorted)
    ax1.set_title('Original Image')
    
    ax2 = plt.subplot(122)
    ax2.imshow(final_img)
    ax2.set_title('Lane Area Annotated')

    return
