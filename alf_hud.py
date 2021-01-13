import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import logging

class HUD:
    '''
    A simulated heads-up-display; composes the final image
    '''
    
    def __init__(self):
        
        #--- text formatting params
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.scale = 1
        self.color = [255, 255, 255]
        self.thickness = 3
        
        return

    def blendImages(self, img_undistorted, unwarped_lanes):
        '''
        Annotates original road image with lane area found
        '''
    
        self.img_lane_area = cv2.addWeighted(img_undistorted, 
                                             alpha = 1.0, 
                                             src2  = unwarped_lanes, 
                                             beta  = 0.3, 
                                             gamma = 0.0)
        
        return
    
    def putRadius(self, rad):
        '''
        Draws radius of curvature indicator on image.
        '''
    
        radius = rad
        if radius is None:
            rad_str = "Radius:"
        elif radius >= 2000: 
            # we'll use around 2km for a road that feels straight
            # min curve radius with superelevation 80mph 4575ft ~1.4km
            # per U.S. government specifications for highway curvature: 
            # link: https://tinyurl.com/y42ulukp
            rad_str = "On straightaway"
        else:
            msg = "Radius: {:.2f} km"
            rad_str = msg.format (radius / 1000)
        
        cv2.putText(self.img_lane_area, rad_str, (50, 50), self.font, self.scale, self.color, self.thickness)
        
        return
        
    def putCenterOffset(self, off):
        '''
        Draws radius of curvature indicator on image.
        '''
        
        center_offset = off
        if center_offset is None:
            off_str = "Center offset "
        else:
            if abs (center_offset) < 0.1:  #--- 0.1 m or 10 cm
                #--- if it's within 10cm it's on centerline...comeon!!!!
                off_str = "Roughly on centerline!"
            else:
                if center_offset > 0:
                    msg = "Center Offset: {:.0f} cm Left"
                else:
                    msg = "Center Offset: {:.0f} cm Right"
                off_str = msg.format (abs(center_offset * 100))
        
        cv2.putText(self.img_lane_area, off_str, (50, 100), self.font, self.scale, self.color, self.thickness)
        
        return
    
    def compose(self, img_undistorted, unwarped_lanes, rad, off):
        
        self.blendImages(img_undistorted, unwarped_lanes)
        
        self.putRadius(rad)
        self.putCenterOffset(off)
        
        return self.img_lane_area

    def imshow(self):
    
        plt.figure(figsize=(11, 6.5))
        plt.imshow(self.img_lane_area)
        
        return
