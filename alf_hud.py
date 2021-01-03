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
    
    def __init__(self, advanced_lane_finder, image_warper):
        
        self.alf = advanced_lane_finder
        self.iw  = image_warper
        
        #--- text formatting params
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.scale = 1
        self.color = [255, 255, 255]
        self.thickness = 3
        
        return

    def blendImages(self, img_undistorted):
        '''
        Annotates original road image with lane area found
        '''
    
        lane_area_warped   = self.alf.paintLaneArea()
        lane_area          = self.iw.unwarpPerspective(lane_area_warped)

        self.img_lane_area = cv2.addWeighted(img_undistorted, 
                                             alpha = 1.0, 
                                             src2  = lane_area, 
                                             beta  = 0.3, 
                                             gamma = 0.0)
        
        return
    
    def putRadius(self):
        '''
        Draws radius of curvature indicator on image.
        '''
    
        radius = self.alf.radius()
        if radius is None:
            rad_str = "Radius:"
        else:
            msg = "Radius: {:.2f} km"
            rad_str = msg.format (radius / 1000)
        
        cv2.putText(self.img_lane_area, rad_str, (50, 50), self.font, self.scale, self.color, self.thickness)
        
        return
        
    def putCenterOffset(self):
        '''
        Draws radius of curvature indicator on image.
        '''
        
        center_offset = self.alf.centerOffset()
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
    
    def compose(self, img_undistorted):
        
        self.blendImages(img_undistorted)
        
        self.putRadius()
        self.putCenterOffset()
        
        return self.img_lane_area

    def imshow(self):
    
        plt.figure(figsize=(11, 6.5))
        plt.imshow(self.img_lane_area)
        
        return
