import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import logging

class TopDownWarperCalibrationSet:
    '''
    Data for calculating transformation matrix for an ImageWarper
    
    Methods:
    - __init__
    - showPoints
    
    Notes:
    - Persective transforms using this calibration set result in a top-down view
    - This is the default calibration set for ImageWarper
    '''    
    
    def __init__(self):
        
        #--- save image calibration points were based off of
        self.img = mpimg.imread('test_images/straight_lines1.jpg')
        
        #--- src and dst points for perspective transformations
        #--- coordinages were derived from manual measurement
        self.src_points = np.float32 ([[592, 451], [688, 451],  [1123, 719], [191, 719]])
        self.dst_points = np.float32 ([[291,   0], [1023,   0], [1023, 719], [291, 719]])
        
        '''
        Older src and dest points that worked but not as good as final
        
        self.src_points = np.float32 ([[592, 451], [688, 451], [1123, 719], [191, 719]])
        self.dst_points = np.float32 ([[191,   0], [1123,  0], [1123, 719], [211, 719]])
        
        self.src_points = np.float32 ([[592, 451], [688, 451], [1123, 719], [191, 719]])
        self.dst_points = np.float32 ([[391,   0], [923,   0], [923,  719], [391, 719]])
        '''
        
        return
    
    def showPoints(self):
        '''
        plots src and dst points on image calibration points were based off of
        '''
        
        plt.figure(figsize=(11, 6.5))

        ax = plt.sublot(121)
        for i in range(4):
            ax.plot(src_points[i][0], src_points[i][1], '.')
        return
        ax.imshow(self.img)
        
        ax = plt.sublot(122)
        for i in range(4):
            ax.plot(dst_points[i][0], dst_points[i][1], '.')
        return
        ax.imshow(self.img)
        
    def getSrcPoints(self):
        return self.src_points

        
class ImageWarper:
    '''
    Warps and dewarps images.
    
    Methods:
    - __init__()
    - warpPerspective()
    - unWarpPerspective()

    Notes:
    -  transforms (warps) an image for use by a LaneDetector
    '''

    def __init__(self, calibration_set=None):
        '''
        Calculates the transform and inverse tranformation matrices
        
        Params:
        - calibration set: set of src are dst points of the form [[x1, y2], [x2, y2], ...];
        src points are points on original image; dst points are points that the src points
        will be transformed to
        '''
        
        #--- set default calibration set to top-down perspective
        if calibration_set is None:
            calibration_set = TopDownWarperCalibrationSet()
            
        self.calibration_set = calibration_set
        
        #--- M: transformation matrix
        self.M = cv2.getPerspectiveTransform(calibration_set.src_points, 
                                             calibration_set.dst_points)
        
        #--- use invM when unwarping image
        self.invM = cv2.getPerspectiveTransform(calibration_set.dst_points, 
                                                calibration_set.src_points)
        
        return
        
    def warpPerspective(self, img):
        '''
        warps perspective based on transformation_matrix
        
        Params:
        - img: the image to be distorted; can be RGB or gray scale
        
        Returns:
        - img_warped: the image with the per
        
        Notes:
        - Undistort img with an camera instance before warping
        '''
        
        img_warped = cv2.warpPerspective(img, self.M, 
                                        (img.shape[1], img.shape[0]), 
                                        flags=cv2.INTER_LINEAR)
        
        return img_warped
    
    def unwarpPerspective(self, img):
        
        img_unwarped = cv2.warpPerspective(img, self.invM, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        
        return img_unwarped
        
