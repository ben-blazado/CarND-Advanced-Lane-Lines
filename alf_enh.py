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
    - Sobel_x
    - s_channel
    - enhance: combine Sobel_x gradient and s_channel thresholding
    '''
    
    def __init__(self):
        self.logger = logging.getLogger("Enhancer")
        
        return
    
    def SobelX(self, img, t_min=20, t_max=255):
        '''
        Detects edges by measuring gradients along x axis.
        
        Params:
        - img: an RGB image of road to enhance
        - t_min, t_max: thresholds for gradient along x axis to trigger edge detection
        
        Returns:
        - mask: a binary image highlighting where edges were detected
        '''
        
        msg = "SobelX thresholds {}, {}."
        self.logger.debug(msg.format(t_min, t_max))
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0))
        sobel_scaled = np.uint8(255 * sobel / np.max(sobel))
        
        mask = np.zeros_like(sobel_scaled)
        
        #--- activate (set to "1") all pixels that meet the x gradient thresholds
        mask[(t_min <= sobel_scaled) & (sobel_scaled <= t_max)] = 1
        
        return mask
        
    def HChannel (self, img, t_min=30, t_max=40):
            '''
            Detects pixels meeting h_channel thresholds tuned to yellow.
            
            Params:
            - img: an RGB image of road to enhance
            - t_min, t_max: thresholds for h_channel_channel detection; default threshold values
            were manually chosen for enhancing yellow and white pixels of road lanes
            
            Returns:
            - mask: a binary image highlighting where pixels met the s_channel threshold 
            for yellow and white colors
            '''
            msg = "HChannel thresholds {}, {}."
            self.logger.debug(msg.format(t_min, t_max))
            
            hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            
            #--- extract the s_channel frol hls
            h_channel = hls[:,:,0]
            
            mask = np.zeros_like(h_channel)
            
            #--- activate all pixels that meet the s-channel threshold
            mask[(t_min <= h_channel) & (h_channel <= t_max)] = 1
            
            return mask
        
                
        
    def SChannel (self, img, t_min=50, t_max=255):
        '''
        Detects pixels meeting s_channel thresholds.
        
        Params:
        - img: an RGB image of road to enhance
        - t_min, t_max: thresholds for s_channel detection; default threshold values
        were manually chosen for enhancing yellow and white pixels of road lanes
        
        Returns:
        - mask: a binary image highlighting where pixels met the s_channel threshold 
        for yellow and white colors
        '''
        
        msg = "SChannel thresholds {}, {}."
        self.logger.debug(msg.format(t_min, t_max))
        
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        #--- extract the s_channel from hls
        s_channel = hls[:,:,2]
        
        mask = np.zeros_like(s_channel)
        
        #--- activate all pixels that meet the s-channel threshold
        mask[(t_min <= s_channel) & (s_channel <= t_max)] = 1
        
        return mask
    
    def LChannel (self, img, t_min=202, t_max=255):
        '''
        Detects pixels meeting s_channel thresholds.
        
        Params:
        - img: an RGB image of road to enhance
        - t_min, t_max: thresholds for s_channel detection; default threshold values
        were manually chosen for enhancing yellow and white pixels of road lanes
        
        Returns:
        - mask: a binary image highlighting where pixels met the s_channel threshold 
        for yellow and white colors
        '''
        
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        #--- extract the l_channel frol hls
        l_channel = hls[:,:,1]
        
        mask = np.zeros_like(l_channel)
        
        #--- activate all pixels that meet the s-channel threshold
        mask[(t_min <= l_channel) & (l_channel <= t_max)] = 1
        
        return mask
    
    
    def enhance(self, img):
        '''
        Combines SobelX and SChannel methods for enhancing lanes
        
        Params:
        - img: an RGB image of road to enhance

        Returns:
        - a binary image resulting from bitwise "or" of sobel and s-channel masks
        '''

        #-- mask for edges in shanow
        l_mask_shadow   = self.LChannel(img, 16, 32)
        sobel_mask       = self.SobelX (img, 8, 255)
        mask_edge_shadow = l_mask_shadow & sobel_mask
        
        #--- mask for yellow lanes
        l_mask_yellow = self.LChannel(img, 100, 255)
        s_mask_yellow = self.SChannel(img, 50, 255) 
        mask_yellow = l_mask_yellow & s_mask_yellow
        
        #--- white lane mask
        mask_white = self.LChannel(img)
        
        return mask_edge_shadow | mask_yellow | mask_white
        # return h_channel_mask  | l_channel_mask
        # works:
        # return sobel_mask | h_channel_mask  | l_channel_mask
        
    