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
        
        
    def preprocess__DEV__(self, img):

        def h(deg):
            return deg // 2
            
        def s(pct):
            return np.int8(pct * 255)
            
        def v(pct):
            return np.int8(pct * 255)
    
        #---get shadow
        
        self.logger.debug("Preprocess()")
        
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        hc = hsv[:,:,0]
        sc = hsv[:,:,1]
        vc = hsv[:,:,2]
        # find all pixels in shadow  (black road color)
        v_shadow =  vc <= v(0.25)  # saturation is lower (gray stuff)   32
        # h_shadow = (h(0) <= hc) & (hc <= h(25))
        s_shadow = sc >= s(0.25)
        in_shadow = v_shadow & s_shadow
        # set sat and value of all low val  pixels to 0
        sc[in_shadow] = s(1.0)   
        hc[in_shadow] = h(60)
        vc[in_shadow] = v(0.6)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        

    def maskYellowLane(self, img):
        self.logger.debug("Preprocess()")
        
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask = np.zeros_like(hsv[:,:,1])
        
        mask [(hsv[:,:,1] > 57) & (hsv[:,:,2] > 220)] = 1
        
        return mask
        
    
    

    def preprocess(self, img):
    
        #---isolates yellow in light area!!!
        
        self.logger.debug("Preprocess()")
        
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        s = hsv[:,:,1]
        v = hsv[:,:,2]
        
        # find all pixels with low val  (black road color)
        hi_sat = (s > 57) & (v > 220) # saturation is lower (gray stuff)   32
        # set sat and value of all low val  pixels to 0
        s[~hi_sat] = 0   
        v[~hi_sat] = 0        
        s[hi_sat] = 255   


        #low_val = v < 200   # saturation is lower (gray stuff)   32
        # set sat and value of all low val  pixels to 0
        #s[low_val] = 0   
        #v[low_val] = 0

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
    
    def SobelXMask(self, img, t_min=20, t_max=255):
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
        
        # activate (set to "1") all pixels that meet the x gradient thresholds
        mask[(t_min <= sobel_scaled) & (sobel_scaled <= t_max)] = 1
        
        return mask
        
    def HMask (self, img, t_min=30, t_max=40):
        '''
        Detects pixels meeting h_channel thresholds.
        
        Params:
        - img: an RGB image of road to enhance
        - t_min, t_max: thresholds for h_channel_channel detection; default threshold values
        were manually chosen for enhancing yellow and white pixels of road lanes
        
        Returns:
        - mask: a binary image highlighting where pixels met threshold 
        '''
        msg = "HChannel thresholds {}, {}."
        self.logger.debug(msg.format(t_min, t_max))
        
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        h_channel = hls[:,:,0]
        
        mask = np.zeros_like(h_channel)
        
        # activate all pixels that meet the thresholds
        mask[(t_min <= h_channel) & (h_channel <= t_max)] = 1
        
        return mask
        
                
        
    def SMask (self, img, t_min, t_max):
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
        
        msg = "SMask thresholds {}, {}."
        self.logger.debug(msg.format(t_min, t_max))
        
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        # extract the s_channel from hls
        s_channel = hls[:,:,2]
        
        mask = np.zeros_like(s_channel)
        
        # activate all pixels that meet the s-channel threshold
        mask[(t_min <= s_channel) & (s_channel <= t_max)] = 1
        
        return mask
    
    def LMask (self, img, t_min, t_max):
        '''
        Detects pixels meeting s_channel thresholds.
        
        Params:
        - img: an RGB image of road to enhance
        - t_min, t_max: thresholds for s_channel detection; default 
        threshold values were manually chosen for enhancing yellow and white 
        pixels of road lanes
        
        Returns:
        - mask: a binary image highlighting where pixels met the s_channel threshold 
        for yellow and white colors
        '''
        
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        # extract the l_channel frol hls
        l_channel = hls[:,:,1]
        
        mask = np.zeros_like(l_channel)
        
        # activate all pixels that meet the s-channel threshold
        mask[(t_min <= l_channel) & (l_channel <= t_max)] = 1
        
        return mask
        
        
    def enhance(self, img):
        '''
        Combines SobelX and SMask methods for enhancing lanes
        
        Params:
        - img: an RGB image of road to enhance

        Returns:
        - a binary image resulting from bitwise "or" of sobel and s-channel masks
        '''
        
        sobel_mask       = self.SobelXMask (img, 40, 255)

        mask_yellow = self.maskYellowLane(img) 
        
        mask_white = self.LMask(img, 202, 255)
        
        return sobel_mask | mask_yellow | mask_white
        
        
    def enhance__GOOD__(self, img):
        '''
        Combines SobelX and SMask methods for enhancing lanes
        
        Params:
        - img: an RGB image of road to enhance

        Returns:
        - a binary image resulting from bitwise "or" of sobel and s-channel masks
        '''
        
        sobel_mask       = self.SobelXMask (img, 40, 255)

        pimg = self.preprocess(img)
        mask_yellow = self.SMask(pimg, 200, 255) 
        
        mask_white = self.LMask(img, 202, 255)
        
        return sobel_mask | mask_yellow | mask_white


    # OLD STUFF
        
    def enhance__GOOD__ (self, img):
        '''
        Combines SobelX and SMask methods for enhancing lanes
        
        Params:
        - img: an RGB image of road to enhance

        Returns:
        - a binary image resulting from bitwise "or" of sobel and s-channel masks
        '''
        
        
        #-- mask for edges in shanow
        pimg = self.preprocess(img)
        # pimg = self.preprocessWhite(img)
        # pimg = self.yellowInShadow(img)
        
        # l_mask_shadow   = self.LChannel(pimg, 255, 255)
        sobel_mask       = self.SobelXMask (img, 40, 255)
        #mask_edge_shadow = sobel_mask        
        # mask_edge_shadow = l_mask_shadow & sobel_mask
        #return mask_edge_shadow
        
        # mask for yellow lanes dark
        mask_yellow_pre = self.SMask(pimg, 200, 255) 
        
        # mask for yellow lanes in light areas
        l_mask_yellow = self.LMask(img, 100, 255)
        s_mask_yellow = self.SMask(pimg, 128, 255) 
        mask_yellow = s_mask_yellow # & l_mask_yellow
        
        # white lane mask
        # mask_white = self.LChannel(img, 187, 255)
        # mask_white = self.LChannel(img, 180, 255)
        mask_white = self.LMask(img, 202, 255)
        
        # return mask_edge_shadow | mask_yellow | mask_white
        #return mask_white
        # return mask_yellow_pre | mask_yellow | mask_white
        return sobel_mask | mask_yellow_pre | mask_white # | mask_yellow 
        
        
    def preprocessWhite(self, img):

        
        # set all L havlues of saturated to 0
        self.logger.debug("Preprocess White()")
        
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        #- change hi saturated colors  to white
        in_y = (17 < hls[:,:,0]) & (hls[:,:,0] < 40)
        hi_l =  hls[:,:,1] > 64      # 128
        hi_sat = hls[:,:,2] > 64      # 96
        hls[:,:,1][in_y & hi_sat & hi_l] = 255 
        
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        
        l_thresh = 200
        low_l = hls[:,:,1] < l_thresh   # kinda gray to darker lightness
        hls[:,:,1][low_l] = 0   # set lightness to 0 for lower saturated

        hi_l = hls[:,:,1] > l_thresh
        hls[:,:,1][hi_l] = 255
        
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        
    
        
     
    
