import alf_cam
import alf_enh
import alf_war
import alf_llg
import alf_hud
import logging
import cv2
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
            'w_max_s'
            'w_min_v'
            'max_coeffs'
            'min_samples'
            'N'
            'coeff_bias'
            'clip_start'
            'clip_end'
            'stage'
            'output'
            
        Notes:
        - Default params for project_video.mp4:
        
            project_video = {
                'filename'   : 'project_video.mp4',
                'sob_min_x'  : 40,
                'y_min_s'    : 57,
                'y_min_v'    : 220,
                'w_max_s'    : 25,
                'w_min_v'    : 201,
                'max_coeffs' : 50,
                'min_samples': 3,
                'N'          : 12,
                'coeff_bias' : 0.9,
                'clip_start' : None,
                'clip_end'   : None,
                'stage'      : 5,
                'output'     : 'project_video.mp4'
            }
        '''
    
        self.filename = params['filename']
        
        self.enh.setParams(
                sob_min_x = params['sob_min_x'], 
                y_min_s   = params['y_min_s'], 
                y_min_v   = params['y_min_v'], 
                w_max_s   = params['w_max_s'], 
                w_min_v   = params['w_min_v'])
            
        self.alf.setParams(
                max_coeffs  = params['max_coeffs'], 
                min_samples = params['min_samples'], 
                N           = params['N'], 
                coeff_bias  = params['coeff_bias'])
                
        self.start  = params['clip_start']
        self.end    = params['clip_end']
        self.stage  = params['stage']
        self.output = params['output']
                
        return
    
    def processImg(self, img):
    
        # show search areas (sliding window, linear window)
        # only if stage selected is 3
        srch_only = (self.stage == 3)
        
        img_undistorted     = self.cam.undistort(img)
        binary              = self.enh.enhance(img_undistorted)    
        binary_warped       = self.war.warpPerspective(binary)
        lane_area, rad, off = self.alf.paintLaneArea(binary_warped, srch_only)
        unwarped_lanes      = self.war.unwarpPerspective(lane_area)
        final_img           = self.hud.compose(img_undistorted, 
                                    unwarped_lanes, rad, off)
                                    
        self.frame_number += 1
        pct_complete = int((self.frame_number // self.interval) * 10)
        if (self.frame_number % self.interval) == 0:
            msg = "Video processing in progress: {}% complete."
            self.logger.info (msg.format(pct_complete))
                                    
        
        # change result to one of the outputs above to view
        pipeline = [img_undistorted, binary, binary_warped, lane_area,
                unwarped_lanes, final_img]
        result = pipeline[self.stage]
        
        if len(result.shape) < 3:
            # result is a binary
            return cv2.cvtColor(result * 255, cv2.COLOR_GRAY2RGB)
        else:
            return result
        
        # return final_img
    
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
        self.frame_number = 0
        self.interval = self.num_frames // 10
                    
        processed_clip = clip.fl_image(self.processImg)
        processed_clip.write_videofile(out_vid_filename, logger=None, 
                audio=False)
                
        msg = "Video processing complete: {}."
        self.logger.info(msg.format(out_vid_filename))
        
        return
        
        
        
    
    
    

    
        

    