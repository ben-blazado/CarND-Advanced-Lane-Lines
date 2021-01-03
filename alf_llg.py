import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import logging

#--- image to real space scaling default per lesson Measuring Curvature II
#--- in meters per pixel; used in calculating radius of curvature
#--- as well as offset center
XM_PER_PIXEL = 3.7 / 700    
YM_PER_PIXEL = 30 / 720
    
class Line:
    '''
    A line representing a lane on the road.

    Methods:
    - __init__
    - fit
    - radius: calucaltes radius of curvature
    - generatePoints
    - getPaintPoints
    '''
    
    def __init__(self):

        self.logger = logging.getLogger("Line " + str(id(self))[-4:])

        #--- coefficients of line where x = ay**2 + by + c
        self.a = None
        self.b = None
        self.c = None
        
        #--- list of coefficients of previously found lines
        self.prev_coeffs = []        #--- [[a1, b2, c1], [a2, b2, c2], ...]
        self.max_coeffs = 30    #--- max coeffs to keep before popping oldest coeff


        #--- True if polyfit() solved for line coefficients
        #--- check this variable first before using Line
        self.found = False
        
        #--- numpy array of points representing the line
        #--- these will be generated in call to generateXY
        self.x = None
        self.y = None    #--- should be [(height of image - 1) to 0]

        #--- pts() method combines x and y to form [[x1, y1], [x2, y2], ...]
        #--- used in draw functions
        self.pts = None    

        return
        
    def fit(self, x_points, y_points, polyfit_tries=3):
        '''
        Sets the coefficients of the line through polyfit.
        
        Params:
        - x, y: coordinates of points to fit line
        - real_world: bool. true if trying to fit real world coordinates from image space; used to 
        support calculating radius of curvature in radius(); we should not have to use this since 
        we will try to use the scaling algorithm in "Measuring Curvature II"
        for the coefficients in call to radius()
        - polyfit_tries: number of attempts to polyfit before continuing
        
        Notes:
        - Makes up to default 10 attempts with error checking for a successful fit.
        - Coefficients are set to none if no line found
        - Needed to use error handling for calls to polyfit; would cause "SVD did not 
        converge" error in jupyter notebook every other run; polyfit will likely succeed on second try
        - Discussion of SVD error: https://github.com/numpy/numpy/issues/16744 says its due to windows
        '''
        
        self.pts = None    
        self.x = None
        self.y = None    #--- should be [0 to (height of image - 1)]
        
        self.found = False         # true if polyfit succesfully fits the line
        tries = 0
        while tries < polyfit_tries and not self.found:
            tries += 1
            try:
                #--- we solve for X!!! i.e. x = ay**2 + by + c
                fit = np.polyfit(y_points, x_points, deg=2)
                if tries > 1:
                    msg = "Polyfit succeeded on try {}."
                    self.logger.warning(msg.format(tries))
                self.found = True
            
            except Exception as e:
                if tries < polyfit_tries:
                    msg = "Polyfit failed: {}. Trying again."
                    self.logger.warning(msg.format(e))
                else:
                    msg = "Polyfit failed to fit a line. {}."
                    self.logger.warning(msg.format(e))
                    
        if self.found:
            self.a = fit[0]    # coeff for y**2
            self.b = fit[1]    # coeff for y
            self.c = fit[2]    # constant
        else:
            self.a = None
            self.b = None
            self.c = None
            
        return
    
    def smooth_dev (self, N=3):
        '''
        Smooths the line by using the average of past coeefficients
        
        Params:
        - N: number of standard deviations from mean coeff value to start smoothing
        
        Returns:
        - self.found: True if a line was found previously in fit and was a good line, or if 
        fit did not find a line by smooth was able to use the average of past coeffs to define 
        a line
        
        Notes:
        - Call after fit!
        - if fit() did not find a line, smooth() can set coeffs to average of coeffs for line!
        thus, finding a line, though it's an average of previous
        
        - Checks if x is within m std devs of the mean of values in arr
        - see ref below for discussion of outlier
        - ref: https://docs.oracle.com/cd/E17236_01/epm.1112/cb_statistical/frameset.htm?ch07s02s10s01.html
        '''
        
        if self.prev_coeffs:
            
            avg_prev_coeffs = np.average(self.prev_coeffs, axis=0)
            
            if self.found:
                
                if len (avg_prev_coeffs) > 10:
                    #--- see if coeff of line is within std devs of avg of previous lines 
                    std_prev_coeffs = np.std(self.prev_coeffs, axis=0)
                    in_range = abs(self.x - avg_prev_coeffs) < N*std_prev_coeffs
                else:
                    in_range = abs(self.x - avg_prev_coeffs) < 50

                if not all(in_range):
                    #--- line is an outlier; smooth it as an average of previous lines
                    self.x = avg_prev_coeffs
                    
                    # self.a, self.b, self.c = np.average (np.vstack((avg_prev_coeffs, np.array([self.a, self.b, self.c]))),                                                         axis = 0)
                else:
                    msg = "Line is good! No need to average."
                    self.logger.debug(msg)
                    
                self.prev_coeffs.append([self.a, self.b, self.c]) 

                if len(self.prev_coeffs) > self.max_coeffs:
                    #--- pop first to remove oldest line from list to prevent being include in average
                    self.prev_coeffs.pop(0)
                    msg = "Coeff buffer full. Removed oldest line. Buffer size: {}."
                    self.logger.debug(msg.format(len(self.prev_coeffs)))
                    
            else:
                #--- line not found, use average of previous coffecients now for the line
                msg = "Line was not found! Using averages of old lines."
                self.logger.debug(msg)
                
                self.a, self.b, self.c = avg_prev_coeffs                
                
                #--- pop first coeff in list to prevents using older coeffs again
                #--- eventually, list decays to nothing if no lines are found
                #--- forcing to not use old information
                self.prev_coeffs.pop(0)
                
                #--- set found to True since we "found" a line 
                #--- using the average of previous coeffs
                self.found = True
                
        elif self.found:
            #--- first coeff to add to list! 
            #--- don't smooth since there is nothing to smooth to
            self.prev_coeffs.append([self.a, self.b, self.c]) 
            
        msg = "Number of old lines/coeffs: {}."
        self.logger.debug(msg.format(len(self.prev_coeffs)))
            
        return self.found
    
    
    def smooth(self, N=3):
        '''
        Smooths the line by using the average of past coeefficients
        
        Params:
        - N: number of standard deviations from mean coeff value to start smoothing
        
        Returns:
        - self.found: True if a line was found previously in fit and was a good line, or if 
        fit did not find a line by smooth was able to use the average of past coeffs to define 
        a line
        
        Notes:
        - Call after fit!
        - if fit() did not find a line, smooth() can set coeffs to average of coeffs for line!
        thus, finding a line, though it's an average of previous
        
        - Checks if x is within m std devs of the mean of values in arr
        - see ref below for discussion of outlier
        - ref: https://docs.oracle.com/cd/E17236_01/epm.1112/cb_statistical/frameset.htm?ch07s02s10s01.html
        '''
        
        return self.found
        
        if self.prev_coeffs:
            
            avg_prev_coeffs = np.average(self.prev_coeffs, axis=0)
            
            if self.found:
                
                #--- see if coeff of line is within std devs of avg of previous lines 
                coeff = np.array([self.a, self.b, self.c])
                std_prev_coeffs = np.std(self.prev_coeffs, axis=0)
                in_range = abs(coeff - avg_prev_coeffs) < N*std_prev_coeffs

                if not all(in_range):
                    #--- line is an outlier; smooth it as an average of previous lines
                    msg = "Line is an outlier! Using averages of old lines."
                    self.logger.debug(msg)
                    
                    msg = "Curr coeffs {}, {}, {}"
                    self.logger.debug(msg.format(self.a, self.b, self.c))

                    msg = "Avg coeffs: {}, {}, {}"
                    self.logger.debug(msg.format(avg_prev_coeffs[0], avg_prev_coeffs[1], avg_prev_coeffs[2]))
                    
                    self.a, self.b, self.c = np.average (
                        np.vstack((avg_prev_coeffs, np.array([self.a, self.b, self.c]))),
                        axis = 0
                    )
                else:
                    msg = "Line is good! No need to average."
                    self.logger.debug(msg)
                    
                self.prev_coeffs.append([self.a, self.b, self.c]) 

                if len(self.prev_coeffs) > self.max_coeffs:
                    #--- pop first to remove oldest line from list to prevent being include in average
                    self.prev_coeffs.pop(0)
                    msg = "Coeff buffer full. Removed oldest line. Buffer size: {}."
                    self.logger.debug(msg.format(len(self.prev_coeffs)))
                    
            else:
                #--- line not found, use average of previous coffecients now for the line
                msg = "Line was not found! Using averages of old lines."
                self.logger.debug(msg)
                
                self.a, self.b, self.c = avg_prev_coeffs                
                
                #--- pop first coeff in list to prevents using older coeffs again
                #--- eventually, list decays to nothing if no lines are found
                #--- forcing to not use old information
                self.prev_coeffs.pop(0)
                
                #--- set found to True since we "found" a line 
                #--- using the average of previous coeffs
                self.found = True
                
        elif self.found:
            #--- first coeff to add to list! 
            #--- don't smooth since there is nothing to smooth to
            self.prev_coeffs.append([self.a, self.b, self.c]) 
            
        msg = "Number of old lines/coeffs: {}."
        self.logger.debug(msg.format(len(self.prev_coeffs)))
            
        return self.found
    
    def generateXY(self, img_ht):
        '''
        Generates the x and y coordinates of the line along the height of the image.
        
        Params:
        - img_ht: height of image
        
        Notes:
        - Call smooth() before calling this function!  M--- wait
        - Updates x_points; array of the x coordinates of the line in int32, none if line was not found
        - Updatesm y_points; array of y coordinates of the line in int32, none if line was not found;
        y points should be from 0 to img_ht - 1
        '''
        
        if not self.found:
            return

        #--- Check if we have already generated lines for this height
        #--- if we havent, generate the points else skip and just reuse what we have already
        #--- generate y points from 0 to img_ht - 1, cast to int32 for ease of plotting
        self.y = np.int32 (np.array([y for y in range(img_ht)]))

        #--- calculate points based on x, cast to int32 for ease of plotting
        self.x = np.int32 (self.a * self.y**2 + self.b * self.y + self.c)

        msg = "Generated points for ht {}."
        self.logger.debug(msg.format(img_ht))
        
        return
    
    def lookupX(self, y):
        '''
        Looks up x-coordinate of line given y-coordinate of line
        
        Params:
        - y: an array of y values 
        
        Returns:
        - x: an array of corresponding x values found by looking up in self.x
        
        Notes:
        - Call generateXY() before calling this
        - Assumes 0 <= y < self.x.size; does not check if y is a valid index!!!
        '''
        
        if not self.found:
            return None
        
        return self.x[y]
    
    def radius(self, y, xm_per_pix=None, ym_per_pix=None):
        '''
        Returns real world radius of curvature at y.
        
        Params:
        - y: y-position in image space; this should be bottom of image; do not convert
        to real world meters; conversion to real world coords is done in function
        - real_world: bool. true if trying to fit real world coordinates from image space
        - ym_per_pix: numer of meters per pixel along y axis; only used if real_world is True
        
        Returns: 
        - R: radius in pixels, or in meters if real_world is True; None if exception
        
        Notes:
        - Call fit() before this function!
        - We use formula in the lesson Measuring Curvature II of Lesson 8:
        "For example, if the parabola is x= a*(y**2) +b*y+c; and mx and my are the scale 
        for the x and y axis, respectively (in meters/pixel); then the scaled parabola is 
        x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+c"; 
        I thinks it's x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+mx*c; anyway so:
           
           real_a = a * mx / my**2 
           
           real_b = b * mx / my
        '''
        
        if not self.found:
            return None
            
        try:
            #--- rescale coeffs and y to real world
            a = self.a * xm_per_pix / ym_per_pix**2
            b = self.b * xm_per_pix / ym_per_pix
            y = y * ym_per_pix    # scale y to real-world!
            R = (1 + (2*a*y + b)**2)**(3/2) / abs(2*a)
            msg = "Radius of curvature: {}."
            self.logger.debug(msg.format(R))
            
        except Exception as e:
            msg = "Error in calculating radius of curvature: {}."
            self.logger.warning(msg.format(e))
            R = None
        
        return R
        
    def baseX(self):
        '''
        Returns the x coordinate of the "base" of the line; i.e. at the bottom of image.
        
        Notes:
        - Used in calculating offset from center for the lane finder
        - Call generateXY() before calling this
        - returns none if no line was found
        '''
        
        if not self.found:
            return None
            
        return self.x[-1]
    
    def getPoints(self):
        '''
        Returns an array of points repesenting the line of the lane for use in cv2 draw functions.
        
        Returns:
        - an array of points like [[[x1, y1], [x2, y2], ...]], shape is like (1, numpoints, 2).

        Notes:
        - Call generateXY() before calling this function
        - Points are suitable for using in cv2 draw functions
        - If no lanes were found, returns None
        - Order of points is from y=0 to number of points
        '''
        
        if not self.found:
            return None
            
        if self.pts is None:
            #--- combine and transform x_ and y_points to [[x1, y1], [x2, y2], ...]
            #--- vstack shape=(2, n), then after transpose shape=(n,2)
            self.pts = np.array([(np.transpose(np.vstack((self.x, self.y))))])
            msg = "New pts calculated."
            self.logger.debug(msg)
        else:
            msg = "Reusing pts."
            self.logger.debug(msg)
        return self.pts
        
    def paint(self, img):
        '''
        Paints line on img.
        '''
        if self.found:
            cv2.polylines(img, 
                          pts       = self.getPoints(), 
                          isClosed  = False, 
                          color     = [255, 0, 0],    #--- default red line color
                          thickness = 20)
        return


class SlidingWindow:
    '''
    Finds lane line points using a sliding window search area.

    Methods:
    - __init__
    - reinit: reinitializes variables so instance can be reused
    - 

    Notes:
    - Lane line points are points on the image that belong to a lane in the road
    '''
    
    def __init__(self, x_mid, target_ht, image_points, 
                 num_segments = 8, x_offset=100, numpoints_found_thresh = 50):

        #--- x1 = x_mid - offset, x2 = x_mid + offset
        self.x_mid    = x_mid
        self.x_offset = x_offset
        self.x1       = x_mid - x_offset
        self.x2       = x_mid + x_offset
        
        self.ht = target_ht // num_segments
        self.y2 = target_ht   #--- y2 is bottom of image
        self.y1 = self.y2 - self.ht
        
        #--- number of points found to calculate
        #--- average x for next mid point when window slides up
        self.numpoints_found_thresh = numpoints_found_thresh
        
        #--- unpack image points which are all the nonzero points in target image
        #--- point (image_points_x[i], image_points_y[i]) is a nonzero point in target image
        self.image_points_x, self.image_points_y = image_points
        
        #--- true if window has slid passed the top of the target image
        self.passed_top = False
        
        #--- holds all image points found in current window using find_points
        self.lane_points_x = []
        self.lane_points_y = []
        
        #--- holds coordinates of windows borders as it is slid up
        #--- used for demo
        self.window_history = []
        
        return
    
    def reinit (self, x_mid, target_ht, image_points, 
                 num_segments = 8, x_offset=100, numpoints_found_thresh = 50):
        '''
        Reinitializes the variables to reuse the instace. Simply calls __init__.
        '''
        
        self.__init__(x_mid, target_ht, image_points, 
                 num_segments, x_offset, numpoints_found_thresh)
        return
    
    def findPoints(self):
        '''
        Finds points in image_points that are in window.
        
        Returns:
        - (lane_points_x, lane_points_y): a tuple of all the x points and y points of the image
        that belong to the lane; so that the following is a point in the image 
        that belongs to the lane (lane_points_x[i], lane_points_y[i]); the points can then be used
        in poly_fit to estimate the lane line (i.e. polyfit (y_lane_points, x_lane_points))
        
        Notes:
        - Accumulates points found in lane_points_x and _y
        - Calls slideUp() at the end to prepare the window to detect the next set of points
        - mid_x updated to average if the number of points found exceed threshold
        '''
        
        while not self.passed_top:        
            
            self.window_history.append([self.x1, self.y1, self.x2, self.y2])
            
            #--- mask in all points within window by x and y values
            x_bool_mask = (self.x1 <= self.image_points_x) & (self.image_points_x <= self.x2)
            y_bool_mask = (self.y1 <= self.image_points_y) & (self.image_points_y <= self.y2)

            #--- bit wise the x and y masks to get the actual points
            xy_bool_mask = (x_bool_mask) & (y_bool_mask)

            #--- apply mask to image_points_x and _y to find the points that are in window region
            points_found_x = self.image_points_x[xy_bool_mask]
            points_found_y = self.image_points_y[xy_bool_mask]

            #--- collect the points found into lane_points_x and _y
            #--- if window is not slid up after call to findpoints, 
            #--- then lane_points may contain duplicate points on next call to findpoints
            self.lane_points_x.extend(points_found_x)
            self.lane_points_y.extend(points_found_y)

            #--- update the midpoint if enough points found above threshold
            if len(points_found_x) >= self.numpoints_found_thresh:
                #--- must be INT! or 
                self.x_mid = np.int(np.average(points_found_x))
                
            self.slideUp()
        
        return (self.lane_points_x, self.lane_points_y)
    
    def slideUp (self):
        '''
        Updates window position by decreasing y1 and y2
        
        - y2 updated first to one line above y1
        - y1 is then updated to y2 - height of window
        - if y2 <= 0, then window has reached the top
        - x1 and x2 are updated in case find_points updated xmid
        '''
        
        #--- update y2 (bottom) of window to line above top
        self.y2 = self.y1 - 1
        
        if self.y2 > 0:
            self.y1 = self.y1 - self.ht - 1
            if self.y1 < 0:
                #--- set y1 to 0 if y1 is below 0
                self.y1 = 0

            self.x1 = self.x_mid - self.x_offset
            self.x2 = self.x_mid + self.x_offset

        else:
            self.passed_top = True
            
        return
    
    
class LinearWindow:
    '''
    Find points of a road lane in an image that are within a certain offset of a line.
    
    Methods:
    - __init__
    - reinit
    - findpoints
    
    Notes:
    - A lane line found in a previous frame by SlidingWindow or LinearWindow is used
    to start the search.
    '''
    
    def __init__ (self, line, target_ht, image_points, x_offset=25):
        '''
        Defines the boarders of the linear search area along line.
        
        Params:
        - line: a line found a previous frame of the road
        - target_ht: height of the road image being examined
        - image_points: all nonzero points of the image if the form [[x1, x2, ...], [y1, y2, ...]]
        - x_offset: the offset along the x-axis of line to determine search area
        '''
        
        self.line = line
        self.ht = target_ht
        self.image_points_x, self.image_points_y = image_points
        
        #--- x1 and x2 define the borders/extents of the search area about the line
        #--- use line.X() to generate the x points of the border
        self.x1 = line.lookupX(self.image_points_y) - x_offset
        #--- add offset twice to x1 to get right border of the search area
        self.x2 = self.x1 + x_offset + x_offset
        
        #--- lane points have not been found yet
        self.lane_points_x = []
        self.lane_points_y = []
        
        return
    
    def reinit (self, line, target_ht, image_points, x_offset=100):
        '''Reinitializes to reuse instance.'''
        
        self.__init__(line, target_ht, image_points, x_offset)
        
        return
    
    def findPoints (self):
        '''
        Collects all the points of the image within the linear border.
        
        Returns:
        - lane_points_x, lane_points_y: points on the road image that are part of the lane;
        these lane points are of the for [x1, x2, ...] and [y1, y2, ...] and are
        fed to polyfit to determine the line that fits the area define by the lane points.
        '''
        
        #--- mask in all points within linear window by x values
        x_bool_mask = (self.x1 <= self.image_points_x) & (self.image_points_x <= self.x2)
        
        #--- apply mask to image_points_x and _y to find the points 
        #--- that are in linear window search area 
        points_found_x = self.image_points_x[x_bool_mask]
        points_found_y = self.image_points_y[x_bool_mask]
        
        #--- collect the points found into lane_points_x and _y
        #--- if window is not slid up after call to findpoints, 
        #--- then lane_points may contain duplicate points on next call to findpoints
        self.lane_points_x.extend(points_found_x)
        self.lane_points_y.extend(points_found_y)
        
        return self.lane_points_x, self.lane_points_y
    

class LaneLineFinder:
    """
    Finds a line representing the lane of a road.
    
    Methods:
    - __init__
    - updateXStart: need to implement in Child Class
    - findImagePoints
    - getSlidingWindow
    - getLinearWindow
    - selectLanePointsFinder
    - checkLaneLine
    - getPaintPoints
    - paint
    """
    
    def __init__(self, buffer_size=180):
        '''
        Initlalizes LaneLineFinder.
        
        Params:
        - buffer_size: number of a, b, c line-coefficients to keep; default to 180 
        for 3 secs of video assuming 60fps; 6 secs of video if 30fps
        '''
        
        self.logger = logging.getLogger("LaneLineFinder " + str(id(self))[-4:])
        
        #--- topdown image of the road in binary format; should be fed in from ImageWarper
        self.binary_warped = None
        
        #--- x value where sliding window starts its search
        self.x_start       = None
        
        #--- nonzero points of road image
        self.image_points  = None
        
        #--- holds lane line used to manage math of the line
        self.lane_line = Line()
        
        #--- variables for the lane search methods
        self.sliding_window     = None 
        self.linear_window      = None
        self.lane_points_finder = None    #--- will eirher be sliding_window or linear_window
        
        #--- helps to select lane points search algorithm 
        self.use_linear_window  = False    #--- If true, use linear search window method, else use sliding
        
        #--- holds buffer the coefficients of the last buffer_size lines
        #--- used for averaging out lines and other metrics
        self.buffer_size = buffer_size     #--- number of measurements to hold in each array
        self.a = np.array([])
        self.b = np.array([])
        self.c = np.array([])
        
        #--- points of the lane line in ([[x1, y1], [x2, y2], ...) format that is used in cv2 drawing functions
        self.paint_points = None
        
        return
    
    def updateXStart(self):
        '''
        Set that value of x_start used to begin the search using slidingWindow.
        
        Note:
        - The method for setting the x_start depends on left or right sidedness of lane;
        it should be implemented in the child class
        '''
        
        raise NotImplementedError()
        
    def findImagePoints(self):
        '''
        Find all nonzero points on the image
        
        Returns:
        - image_points_x, image_points_y: the x and y coordinates of all the nonzero points of the image;
        the LaneFinder will search through these coordinates to find the points that are part of the lane
        '''
        
        nonzero_points = np.nonzero(self.binary_warped)
        
        #--- y (row) points come first!!! index 0
        #--- x (col) points are in second!!! index 1
        image_points_y = np.array(nonzero_points[0])
        image_points_x = np.array(nonzero_points[1])
        
        self.image_points = (image_points_x, image_points_y)
        
        return 
    
    def getSlidingWindow(self):
        '''
        Returns a sliding window used to search image_points for the points of the lane.
        
        Returns:
        - sliding_window
        
        Notes:
        - If a sliding window instance already exists, it is reinitialized and reused.
        - The starting X position must always be calculated and set for the sliding window.
        '''

        self.updateXStart()
        if self.sliding_window:
            #--- resuse existing and reinitizalize instance
            self.sliding_window.reinit(self.x_start,
                                      target_ht    = self.binary_warped.shape[0],
                                      image_points = self.image_points)
        else:
            self.sliding_window = SlidingWindow(self.x_start,
                                               target_ht    = self.binary_warped.shape[0],
                                               image_points = self.image_points)
        
        return self.sliding_window
        
    def getLinearWindow(self):
        '''
        Returns a linear_window used to search image_points for the points of the line.
        
        Returns:
        - linear_window
        
        Notes:
        - There must be at least one line detected previously; it is used as the basis
        for the search area for the linear window.
        '''
        
        if self.linear_window:
            #--- reuse and reinitizalize existing instance
            self.linear_window.reinit(self.lane_line, 
                                      target_ht    = self.binary_warped.shape[0],
                                      image_points = self.image_points)
        else:
            self.linear_window = LinearWindow(self.lane_line,
                                              target_ht    = self.binary_warped.shape[0],
                                              image_points = self.image_points)            
        return self.linear_window
        
    def selectLanePointsFinder(self):
        '''
        Selects the approriate LanePointsFinder to search for lane points.
        '''

        if self.use_linear_window:
            #--- use LinearWindow if lane lines were previously found
            self.lane_points_finder = self.getLinearWindow()
        else:
            #--- use sliding window if lane lines were NOT previously found
            self.lane_points_finder = self.getSlidingWindow()
            
        return
    
    def findLaneLine (self, binary_warped):
        '''
        Finds line representing lane of the road using appropriate lane line finder.
        '''
        
        self.binary_warped = binary_warped
        
        self.findImagePoints()
        self.selectLanePointsFinder()
        
        x_lane_points, y_lane_points = self.lane_points_finder.findPoints()
        
        #--- TODO: test for x_lane_points, y_lane_points = None, None
        
        #--- fit the points of the lane onto a line
        self.lane_line.fit(x_lane_points, y_lane_points)

        #--- smooth out the line to reduce frame to frame jitter
        #--- sets use_linear_window to True if smoothing was successful
        self.use_linear_window = self.lane_line.smooth()
        
        #--- generate the points of the line along the image
        self.lane_line.generateXY(img_ht=self.binary_warped.shape[0])
        
        return 
    
    def radius(self):
        '''
        Calculate real world radius of curvature.
        
        Notes:
        - Measurement taken at bottom of image; i.e. img_ht - 1
        '''
        #--- take the radius at the bottom of the image
        y = self.binary_warped.shape[1] - 1
        R = self.lane_line.radius(y, XM_PER_PIXEL, YM_PER_PIXEL)

        msg = "Radius: {}."
        self.logger.debug(msg.format(R))
        return R
        
    def baseX (self):
        '''
        Returns the X coordinate of the lane_line at the bottom of the image.
        
        Notes:
        - Delegates this down to lane_line
        - used to help calculate center offset: center - (left.basex + right.basex / 2)
        '''
        
        return self.lane_line.baseX()
    
    def getPaintPoints(self, flipud=False):
        '''
        Returns an array of points repesenting the line of the lane for use in draw function.
        
        Params:
        - flipud: option for returning the array of points in reverse order.
        
        Returns:
        - an array of points like [[[x1, y1], [x2, y2], ...]], shape is like (1, numpoints, 2).

        Notes:
        - Assume the points have already been generated in call to lane_line.fit()
        - Points are suitable for using in cv2 draw functions
        - If no lanes were found, returns None
        - Set flipud (flip unside down) to true; this is useful for flipping an oppposite lane
        to form a polygon for a fillpoly call
        '''

        self.paint_points = self.lane_line.getPoints()

        #--- flip points upside down if required; useful for right lane in drawing polygon with left lane
        if flipud and self.paint_points is not None:
            #--- if you flipud(paint_points), it will be the same 
            #--- since there is only 1 element in first dimension
            #--- first element of paint_points contains the actual points
            #--- so flipud (paint_points[0]) then apply np.array([])
            return np.array([np.flipud(self.paint_points[0])])
        
        else:
            return self.paint_points
    
    def paint(self, img):
        '''
        Paints the points of the line on canvas
        
        Params:
        - img: an RGB image to draw the lane line on.
        '''
        msg = "Painting lane."
        self.logger.debug(msg)
        self.lane_line.paint(img)

        return
    
        
class LeftLaneLineFinder(LaneLineFinder):
    '''
    LaneLineFinder for left lane.
    
    Methods:
    - updateXStart: implements base to set the x_start of the LaneLineGenerator
    '''

    def updateXStart(self):
        
        #--- get bottom half of image for histogram
        bottom_half = self.binary_warped[self.binary_warped.shape[0] // 2:,:]
        histogram = np.sum(bottom_half, axis=0)

        #--- search only **left** half of histogram
        mid_x = histogram.shape[0] // 2
        self.x_start = np.argmax(histogram[:mid_x])
        
        return
    
    
class RightLaneLineFinder(LaneLineFinder):
    '''
    LaneLineFinder for right lane.
    
    Methods:
    - updateXStart: implements base to set the x_start of the LaneLineGenerator
    '''
    
    def updateXStart(self):
        
        bottom_half = self.binary_warped[self.binary_warped.shape[0] // 2:,:]
        histogram = np.sum(bottom_half, axis=0)

        #--- search only **right** half of histogram
        mid_x = histogram.shape[0] // 2
        self.x_start = mid_x + np.argmax(histogram[mid_x:])
        
        return
    
    
class AdvancedLaneFinder:
    '''
    Manages a left and right lane finder to find lanes in a road image
    
    Methods:
    - __init__
    - findLanes
    - radius 
    - centerOffset
    - paintLaneArea
    - 
    '''
    
    def __init__(self):
        
        #--- TODO: calculate fps
        
        self.logger = logging.getLogger("AdvancedLaneFinder")
        
        self.left_lane_line_finder  = LeftLaneLineFinder()
        self.right_lane_line_finder = RightLaneLineFinder()
        
        self.avg_radius = None

        self.binary_warped = None
        
        #--- middle x coordinate of image; used for calculating center offset
        self.mid           = None
        self.center_offset = None
        
        return
    
    def findLanes(self, binary_warped):
        
        msg = "---------------------------"
        self.logger.debug(msg)
        msg = "Finding lanes in new frame."
        self.logger.debug(msg)
        msg = "---------------------------"
        self.logger.debug(msg)
        
        self.binary_warped = binary_warped
        
        self.left_lane_line_finder.findLaneLine(binary_warped)
        self.right_lane_line_finder.findLaneLine(binary_warped)
        
        return
    
    def radius(self):
        '''
        Caculates the radius of curvature as the average reported from left and right lanes
        '''

        t = 0    #--- sums up radii values
        n = 0    #--- count of number of samples
        
        radius_left = self.left_lane_line_finder.radius()
        t  += radius_left if radius_left is not None else 0
        n  += 1 if radius_left is not None else 0
        
        radius_right = self.right_lane_line_finder.radius()
        t  += radius_right if radius_right is not None else 0
        n  += 1 if radius_right is not None else 0
        
        msg = "Radius left and right: {}, {}"
        self.logger.debug(msg.format(radius_left, radius_right))
        if n > 0:
            self.avg_radius = t / n
        else:
            self.avg_radius = None

        msg = "AVg Radius: {}."
        self.logger.debug(msg.format(self.avg_radius))
            
        return self.avg_radius
    
    def centerOffset(self):
        '''
        Caclulates the offset from center in real world meters
        
        '''
        
        img_ctr = self.binary_warped.shape[1] // 2
        
        t = 0    #--- sums up x values
        n = 0    #--- count of number of samples

        #--- get distance from center
        x_left =  self.left_lane_line_finder.baseX()
        t  += x_left if x_left is not None else 0
        n  += 1 if x_left is not None else 0
        
        x_right = self.right_lane_line_finder.baseX()
        t  += x_right if x_right is not None else 0
        n  += 1 if x_right is not None else 0
        
        if n > 0:
            lane_center = t / n
            self.center_offset = (lane_center - img_ctr) * XM_PER_PIXEL
        else:
            self.center_offset = None
            
        msg = "Center offset: {}."
        self.logger.debug(msg.format(self.center_offset * 100))
            
        return self.center_offset 
        
    def paintLaneArea(self):
        '''
        Paints the lane and area between lane onto an image
        '''
        
        #---
        #--- create the blank image for painting
        #---
        
        #--- black screen [0 0 0 ...]
        img_binary = np.zeros_like(self.binary_warped).astype(np.uint8)
        
        #--- black RGB img [[0 0 0], [0 0 0], [0 0 0], ...]
        img = np.dstack((img_binary, img_binary, img_binary))
        
        #---
        #--- paint lanes
        #---
        self.left_lane_line_finder.paint(img)
        self.right_lane_line_finder.paint(img)
        
        #---
        #--- paint area between lanes
        #---
        
        #--- get polygon for lane area by getting paint points of each lane finder
        left_lane_paint_points = self.left_lane_line_finder.getPaintPoints()
        
        #--- need to "reverse" right points array so tail of pts_left is next to head of pts_right
        #--- this ordering allows fillpoly to traverse around perimeter, 
        #--- if the order of points is not flipud, the fill will look like a bowtie
        right_lane_paint_points = self.right_lane_line_finder.getPaintPoints(flipud=True)
        
        if left_lane_paint_points is not None and right_lane_paint_points is not None:
            #--- combaine paintpoints into a polygon
            lane_area_paint_pts = np.hstack((left_lane_paint_points, right_lane_paint_points))

            #--- paint the lane area
            cv2.fillPoly(img, lane_area_paint_pts, [0,255,0])
            
        return img
        