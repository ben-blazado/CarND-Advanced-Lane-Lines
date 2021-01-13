import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import logging

class Enhancer:
    '''
    Enhances appearance of road lanes using edge detection and color channels.
    
    Methods:
    - __init__
    - laneMask
    - sobelXMask
    - enhance: combines Sobel_x and lane masks
    '''
    
    def __init__(self):
        '''
        Notes:
        - Params for project_video.mp4: sob_min_x=40, y_min_s=57, 
        y_min_v=220, w_min_v=201
        
        '''
        
        self.logger = logging.getLogger("Enhancer")
        
        # min threshold for sobel x gradient
        self.sob_min_x = None 
        
        # min thresh for s and v channels for yellow lanes
        self.y_min_s   = None 
        self.y_min_v   = None
        
        # min thresh for v channel for white lane
        self.w_min_v   = None
        
        return
        
    def setParams(self, sob_min_x=40, y_min_s=57, y_min_v=220, 
             w_max_s=25, w_min_v=201):
        '''
        Sets thresholds for masks.
        
        Params:
        - sob_min_x: min threshold for sobel x gradient
        - y_min_s, y_min_v: min thresh for s and v channels for yellow lanes
        - w_min_v: min thresh for v channel for white lane
        '''
            
        self.sob_min_x = sob_min_x
        self.y_min_s = y_min_s
        self.y_min_v = y_min_v
        self.w_min_v = w_min_v
        self.w_max_s = w_max_s
        
        return
        
    def laneMask(self, img):
        '''
        Returns a mask for the yellow and white lanes from a road image.
        
        Params:
        - img: RGB image of a road with lanes
        
        Returns:
        - Binary mask of yellow and white lanes
        
        '''
        self.logger.debug("laneMask()")
        
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        h = hsv[:,:,0] # h-channel
        s = hsv[:,:,1] # s-channel
        v = hsv[:,:,2] # v-channel
        
        # mask for yellow lane
        y_mask = np.zeros_like(s)
        y_h = (10 <= h) & (h <= 25)
        y_s = s > self.y_min_s
        y_v = v > self.y_min_v
        y_mask [(y_h & y_s & y_v)] = 1
            
        # mask for white lane
        w_mask = np.zeros_like(s)
        w_s = s < self.w_max_s
        w_v = self.w_min_v < v
        w_mask[(w_s & w_v)] = 1
        
        return y_mask | w_mask
    
    def sobelXMask(self, img):
        '''
        Detects edges by measuring gradients along x axis.
        
        Params:
        - img: an RGB image of road to enhance
        
        Returns:
        - mask: a binary image highlighting where edges were detected
        '''
        
        self.logger.debug("sobelXMask()")
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0))
        sobel_scaled = np.uint8(255 * sobel / np.max(sobel))
        
        mask = np.zeros_like(sobel_scaled)
        
        # activate (set to "1") all pixels that meet the x gradient thresholds
        mask[(self.sob_min_x <= sobel_scaled) & (sobel_scaled <= 255)] = 1
        
        return mask
        
    def enhance(self, img):
        '''
        Combines SobelX and SMask methods for enhancing lanes
        
        Params:
        - img: an RGB image of road to enhance

        Returns:
        - a binary image resulting from bitwise "or" of sobel and s-channel masks
        '''
        
        sobel_mask = self.sobelXMask (img)

        lane_mask = self.laneMask(img) 
        
        return sobel_mask & lane_mask
