'''
Demonstration functions for use in jupyter notebook
'''
import alf_cam
import alf_enh
import alf_war
import alf_llg
import matplotlib.pyplot as plt


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
    
    plt.figure(figsize=(14,6))
    ax1 = plt.subplot(121)
    ax1.imshow(img)
    ax1.set_title('Original Image with Barrel Distortion')
    ax2 = plt.subplot(122)
    ax2.imshow(img_undist)
    ax2.set_title('Corrected Image after Calibration')

    return
    
    
#
# Demonstration functions
#

def demoWarpImage(img):

    cam = alf_cam.Camera()
    cam.calibrate()
    img_undist = cam.undistort(img)
    
    war = alf_war.ImageWarper()
    war.calibrate()
    
    img_orig = np.copy(img_undist)
    img_orig = war.drawSrc(img_orig)
    
    img_warped = war.warpPerspective(img_undist)
    img_warped = war.drawDst(img_warped)
    
    plt.figure(figsize=(14,6))
    
    ax1 = plt.subplot(121)
    ax1.imshow(img_orig)
    ax1.set_title('Original Image')
    
    ax2 = plt.subplot(122)
    ax2.imshow(img_warped)
    ax2.set_title('Warped Imaged')
    
    return
    
    

def demoLaneSearch(img):

    cam  = alf_cam.Camera()
    cam.calibrate()
    
    enh  = alf_enh.Enhancer()
    enh.setParams(40, 57, 220, 201)
    
    war  = alf_war.ImageWarper()
    war.calibrate()
    
    alf  = alf_llg.AdvancedLaneFinder()    
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