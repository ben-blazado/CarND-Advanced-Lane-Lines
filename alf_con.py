import alf_cam
import alf_enh
import alf_war
import alf_llg
import alf_hud
import logging
from moviepy.editor import VideoFileClip

class Controller:
    
    def __init__(self):
    
        self.logger = logging.getLogger("Controller")
        
        self.logger.info("Initializing components...")
        
        self.cam  = alf_cam.Camera()
        self.enh  = alf_enh.Enhancer()
        self.war  = alf_war.ImageWarper()
        self.alf  = alf_llg.AdvancedLaneFinder()
        self.hud  = alf_hud.HUD()
        
        self.cam.calibrate()
        self.war.calibrate()
        
        self.num_frames   = None
        self.pct_complete = None
        self.frame_number = None
        self.interval     = None
        
        self.filename = None
        self.start    = None
        self.end      = None
        
        self.logger.info("...components initialized.")
        
        return
    
    def setParams(self, params):
        '''
        Sets tunable parameters for pipeline.
        
        Params:
        - params: a dict that the following keys used to set parameters
        for enhancer and advanced lane finder:
            'sob_min_x'
            'y_min_s'
            'y_min_v'
            'w_min_v'
            'max_coeffs'
            'min_samples'
            'N'
            'coeff_bias'
        '''
    
        self.filename = params['filename']
        
        self.enh.setParams(
                sob_min_x = params['sob_min_x'], 
                y_min_s   = params['y_min_s'], 
                y_min_v   = params['y_min_v'], 
                w_min_v   = params['w_min_v'])
            
        self.alf.setParams(
                max_coeffs  = params['max_coeffs'], 
                min_samples = params['min_samples'], 
                N           = params['N'], 
                coeff_bias  = params['coeff_bias'])
                
        self.start  = params['clip_start']
        self.end    = params['clip_end']
        self.output = params['output']
                
        return
    
    def processImg(self, img):
    
        img_undistorted     = self.cam.undistort(img)
        binary              = self.enh.enhance(img_undistorted)    
        binary_warped       = self.war.warpPerspective(binary)
        lane_area, rad, off = self.alf.paintLaneArea(binary_warped)
        unwarped_lanes      = self.war.unwarpPerspective(lane_area)
        final_img           = self.hud.compose(img_undistorted, 
                                    unwarped_lanes, rad, off)
                                    
        self.frame_number += 1
        pct_complete = int((self.frame_number/self.num_frames) * 100)
        if (self.frame_number % self.interval) == 0:
            msg = "Video processing in progress: {}% complete."
            self.logger.info (msg.format(pct_complete))
                                    
        return final_img
    
    def processVideo(self, params):
    
        self.setParams(params)
    
        msg = "Processing video: {}..."
        self.logger.info(msg.format(self.filename))
    
        out_vid_filename = "output_images/" + self.output
    
        if self.start is None:
            clip = VideoFileClip(self.filename)
        else:
            clip = VideoFileClip(self.filename).subclip(t_start=self.start, 
                    t_end=self.end)
                    
        self.num_frames = clip.fps * clip.duration
        self.pct_complete = 0
        self.frame_number = 0
        self.interval = self.num_frames // 10
                    
        processed_clip = clip.fl_image(self.processImg)
        processed_clip.write_videofile(out_vid_filename, logger=None, 
                audio=False)
                
        msg = "Video processing complete: {}."
        self.logger.info(msg.format(out_vid_filename))
        
        return
        
        
        
    
    
    

    
        

    