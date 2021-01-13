import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import logging
# %matplotlib inline

class ChessboardImage:
    '''
    An image of a chessboard used to calibrate Camera.
    
    Methods:
    - __init__: specify base, filename, and inner square dims of chessboad image.
    - findChessboardCorners
    
    Notes:
    - objpoints and imgpoints are used to calibrate camera.
    - img_corners can be used to display the image to verify that corners were found correctly.
    '''
    
    def __init__ (self, base_path, filename, inner_dims):
        self.logger = logging.getLogger ("ChessboardImage")
        
        #--- RGB image of chessboard
        self.img = mpimg.imread(base_path + filename)
        self.filename = filename
        
        #---inner dimensions of chessboard
        self.xdim, self.ydim = inner_dims

        #--- object and image points used in findChessboardCorners()
        self.objpoints = None
        self.imgpoints = None
        
        #--- image of chessboard annotated with corners found
        self.img_corners = None
        
        msg = 'Image loaded for {} with dims {}.'
        self.logger.debug (msg.format(self.filename, (self.xdim, self.ydim)))
        
        return
    
    def findChessboardCorners(self):
        '''
        Finds the chessboard corners of the chessboard image.
        
        Returns:
        - corners_found: the image space coordinates of the corners of the chessboard image.
        
        '''
        
        #--- given the chessboard image and inner dimensions (xdim, ydim), find the chessboard corners 
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        corners_found, corners = cv2.findChessboardCorners(gray, (self.xdim, self.ydim), flags=None)
        
        if corners_found:
            #--- generate object points in the XY plane; Z=0
            #--- dimensions will be based on the chessboard_images
            #--- corners found are placed in an order from left to right, top to bottom
            #--- thereforem object points are generated in column (X) priority order
            #--- ref: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
            #--- object_points should be: (0,0,0), (1,0,0), (2,0,0)..(xdim-1,0,0), (1,1,0)...
            self.objpoints = np.zeros(shape=(self.xdim * self.ydim, 3), dtype=np.float32)
            self.objpoints[:, :2] = np.array([(x, y) for y in range(self.ydim) for x in range(self.xdim)])
            
            #--- save corners found to imgpoints
            self.imgpoints = corners
            
            #--- save image with corners annotated to img_corners
            self.img_corners = np.copy (self.img)
            self.img_corners = cv2.drawChessboardCorners(self.img_corners, (self.xdim, self.ydim), self.imgpoints, True)                 
            
            msg = "Found corners for {}."
            self.logger.debug (msg.format(self.filename))
        else:

            msg = "DID NOT FIND corners for {}, dims: {}."
            self.logger.warning (msg.format(self.filename, (self.xdims, self.ydims)))
            
        return corners_found
    
    
class ChessboardCameraCalibrationSet:
    '''
    Distortion calibration set for Camera.
    
    Methods:
    - __init__: specifies path, filenames and inner square dimensions chessboard images.
    - loadChessboardImages
    - findChessboardCorners
    - showCorners
    - getCalibrationParams
    - getChessboardImg
    '''
    
    def __init__(self):
        
        self.logger = logging.getLogger("ChessboardImages")
        
        #--- chessboard image meta data (basepath, filenames with inner dims, image shape)
        #--- base path to images
        self.basepath = 'camera_cal/'
        #--- image filenames with corresponding inner dimensions manually counted
        self.filenames_dims = [
            ('calibration1.jpg', (9,5)),
            ('calibration2.jpg', (9,6)),
            ('calibration3.jpg', (9,6)),
            ('calibration4.jpg', (5,6)),  #--- close corners, watch it
            ('calibration5.jpg', (7,6)),
            ('calibration6.jpg', (9,6)),
            ('calibration7.jpg', (9,6)),
            ('calibration8.jpg', (9,6)),
            ('calibration9.jpg', (9,6)),
            ('calibration10.jpg', (9,6)),
            ('calibration11.jpg', (9,6)),
            ('calibration12.jpg', (9,6)),
            ('calibration13.jpg', (9,6)),
            ('calibration14.jpg', (9,6)),
            ('calibration15.jpg', (9,6)),
            ('calibration16.jpg', (9,6)),
            ('calibration17.jpg', (9,6)),
            ('calibration18.jpg', (9,6)),
            ('calibration19.jpg', (9,6)),
            ('calibration20.jpg', (9,6)),
             ]
        
        #--- shape of all images in cols,rows; for later use in calibrate camera
        self.image_shape = (1280, 720)
        
        #--- a list of data about the boards: (RGB image of board, XY inner dimensions, filename)
        self.chessboard_images = []
        
        #--- objpoints and imgpoints used for camera calibration
        #--- will be compiled in findChessboardCorners
        self.objpoints = []
        self.imgpoints = []
        
        return

    def loadChessboardImages(self):
        '''
        Loads chessboard images into self.chessboard_images; 
        call before using find chessboard corners.
        '''
        
        self.logger.info ("Loading chessboard images...")
        
        for filename_dim in self.filenames_dims:
            
            filename = filename_dim[0]
            dims     = filename_dim[1]
            
            #--- read in files
            chessboard_image = ChessboardImage (self.basepath, filename, dims)
            self.chessboard_images.append(chessboard_image)
            
        self.logger.info ("...completed loading chessboard images.")
        
        return
    
    def findChessboardCorners(self):
        '''
        Finds chessboard corners for each chessboard; call loadchessboardimages 
        before this call.
        
        Notes:
        - Calls each chessboard image to find corners
        - Collects chessboard image objpoints and imgpoints to self.objpoints 
        and self.imgpoints
        '''

        self.logger.info ("Finding chessboard corners...")
        
        for chessboard_image in self.chessboard_images:
            if chessboard_image.findChessboardCorners():
                #--- imgpoints[i] corresponds to objpoints[i]
                self.imgpoints.append (chessboard_image.imgpoints)
                self.objpoints.append (chessboard_image.objpoints)
            
        self.logger.info ("...completed finding chessboard corners.")
        
        return
    
    def getCalibrationParams(self):
        '''
        Returns the parameters needed to calibrate Camera.
        
        Returns:
        - objpoints, imgpoints: objpoints and imgpoints generated from 
        findChessboardCorners.
        - image_shape: shape of image of chessboard images
        
        Notes:
        - objpoints, imgpoints, image_shape will be used in camera calibration
        '''
        self.loadChessboardImages()
        self.findChessboardCorners()
        
        return (self.objpoints, self.imgpoints, self.image_shape)
    

    def showCorners(self, ncols=1):
        '''
        Displays images of the chessboard with all chessboard 
        corners annotated.
        
        Params:
        - ncols: number of columns to fit all the images into.
        '''
        
        nimgs = len(self.chessboard_images)
        nrows = nimgs // ncols + (nimgs % ncols)
        plt.figure(figsize=(16, 10 * (nrows + 1) // ncols))
        for idx, chessboard_image in enumerate (self.chessboard_images):
            ax = plt.subplot(nrows, ncols, idx + 1)
            ax.imshow(chessboard_image.img_corners)
            ax.set_title(chessboard_image.filename)
        return
    
    def getChessboardImg(self, idx=0):
        '''
        Returns an image of a chessboard for inspection
        
        Params:
        - idx: index of chessboard image to inspect
        '''
        
        return self.chessboard_images[idx].img
    

class Camera:
    '''
    Creates image corrected for distortion.
    
    Methods:
    - __init__
    - calibrateCamera: call before using undistort().
    - undistort
    '''
    
    
    def __init__ (self):
        
        self.logger = logging.getLogger("Camera")
        
        #--- camera matrix
        self.mtx = None
        
        #--- distortion coefficient
        self.dist = None
        
        #--- image shape in X,Y (numcols, numrows)
        self.image_shape = None
        
        return
    

    def calibrate(self, calibration_set=None):
        '''
        Calculates the camera matrix and distortion coefficients 
        used to correct distortion.
        
        Params:
        - calibration_set: a class with a function called getCalibrationParams 
        that returns objpoints, imgpoints, and image_shape used for camera 
        calibration.
        '''
        
        self.logger.info("Calibrating...")
        
        if calibration_set is None:
            calibration_set = ChessboardCameraCalibrationSet()
            
        objpoints, imgpoints, self.image_shape = calibration_set.getCalibrationParams ()
        
        #--- rotation and translation vectors not used for this project
        cal_found, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, 
                imgpoints, self.image_shape, None, None)
        
        if cal_found:
            self.logger.info("...calibrated.")            
        else:
            self.logger.warning("...calibration failed.")            
        
        return
        
    def undistort(self, img):
        '''
        corrects image distortion
        
        Paramters:
        - img: image to apply distortion correction
        
        Returns:
        - img_undist: the undistorted image
        
        Notes:
        - img_undist should then be fed to the LaneEnhancer 
        '''
        img_undist = cv2.undistort(img, self.mtx, self.dist)
        
        return img_undist
        
    def stretch(self, img):
        '''
        Experimental only. Stretches/normalizes histogram for s and v channels
        
        Notes:
        - Not used for this project.
        '''
    
        def stretchChan(c):
            avg = np.average(c)
            lo = c.min()
            hi = c.max()

            below = c[c < avg]
            below = 128 - 128 * (avg - below)/(avg - lo)
            c[c < avg] = np.clip(below, 0, 255)

            above = c[c >= avg]
            above = 128 + 128 * (above - avg)/(hi - avg)
            c[c >= avg] = np.clip(above, 0, 255)
            
            return
    
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        ht = hsv.shape[0]
        mid = ht // 2

        s = hsv[mid:ht,:,1] # s-channel
        v = hsv[mid:ht,:,2] # v-channel
        
        stretchChan(s)
        stretchChan(v)
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return rgb
    

