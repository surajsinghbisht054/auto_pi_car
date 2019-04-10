#!/usr/bin/python
#
#
# =++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#       ____     _    __     ____                                __     _             ____       
#      / __ )   (_)  / /_   / __/  ____    _____  ___    _____  / /_   (_)   ____    / __/  ____ 
#     / __  |  / /  / __/  / /_   / __ \  / ___/ / _ \  / ___/ / __/  / /   / __ \  / /_   / __ \
#    / /_/ /  / /  / /_   / __/  / /_/ / / /    /  __/ (__  ) / /_   / /   / / / / / __/  / /_/ /
#   /_____/  /_/   \__/  /_/     \____/ /_/     \___/ /____/  \__/  /_/   /_/ /_/ /_/     \____/ 
#                                                                                              
# =++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                                     
#
#                                                                          Suraj Singh Bisht
#                                                                          surajsinghbisht054@gmail.com
#
# 
#
#           This Script Is A Part Of RC-AIA Framework.
#           Created For Educational And Practise purpose Only.
#			Please Don't Remove Author Initials.
#			Inspired By Udacity Self Driving Car Enginner Nanodegree.
#
#

__author__ 	    = "Suraj Singh Bisht"
__description__ = "Auto-Pilot Rasberry Pi Car Project"
__version__	    = "Beta"
__date__	    = "SEPT, 2018"
__email__	    = "surajsinghbisht054@gmail.com"


# import modules
import random
from config import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import collections
import warnings
from steering import PID
warnings.simplefilter('ignore', np.RankWarning)

# our driving wheel
pid = PID(0.01, 0.001, 1.0)

ym_per_pix = 30.0 / 720.0   # meters per pixel in y dimension
xm_per_pix = 3.7 / 700.0  # meters per pixel in x dimension

def resize(frame):
	return cv2.resize(frame, (WIDTH, HEIGHT))

# Cutting region
def FocusRegion(img, x1=cx1, y1=cy1, x2=cx2, y2=cy2):
	img = img[y1:y2, x1:x2]
	#return cv2.resize(img, (WIDTH, HEIGHT))
	return img

# Get Random Color
def randomcolor():
	'''Get Random Color Tuples'''
	r  = random.randint(0, 255)
	g  = random.randint(0, 255)
	b  = random.randint(0, 255)
	return (r,g,b)


# Show Images
def imshow(title, frame):
	if GTK:

		# print '[+] Initialising GTK Settings'
		cv2.imshow(title, frame)
		if cv2.waitKey(1)==27:
			return True
		else:
			return False
	else:
		#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		print '[+] Initialising matplotlib settings'
		b,g,r = cv2.split(frame)
		frame = cv2.merge([r,g,b])
		plt.imshow(frame)
		plt.show()
		raw_input('[+] Press Enter To Exit..')
		return True

	return 



class Line:
	"""
	Class to model a lane-line.
	"""
	def __init__(self, buffer_len=10):

		# flag to mark if the line was detected the last iteration
		self.detected = False

		# left lane x,y coordinates
		self._x_coordinate = None
		self._y_coordinate = None

		# polynomial coefficients fitted on the last iteration
		self.last_fit_pixel = None
		self.last_fit_meter = None

		# list of polynomial coefficients of the last N iterations
		self.recent_fits_pixel = collections.deque(maxlen=buffer_len)
		self.recent_fits_meter = collections.deque(maxlen=2 * buffer_len)

		self.radius_of_curvature = None

		# store all pixels coords (x, y) of line detected
		self.all_x = None
		self.all_y = None

	def update_line(self, new_fit_pixel, new_fit_meter, detected, clear_buffer=False):
		"""
		Update Line with new fitted coefficients.

		:param new_fit_pixel: new polynomial coefficients (pixel)
		:param new_fit_meter: new polynomial coefficients (meter)
		:param detected: if the Line was detected or inferred
		:param clear_buffer: if True, reset state
		:return: None
		"""
		self.detected = detected

		if clear_buffer:
			self.recent_fits_pixel = []
			self.recent_fits_meter = []

		self.last_fit_pixel = new_fit_pixel
		self.last_fit_meter = new_fit_meter

		self.recent_fits_pixel.append(self.last_fit_pixel)
		self.recent_fits_meter.append(self.last_fit_meter)

	def draw(self, mask, color=(255, 0, 0), line_width=50, average=False):
		"""
		Draw the line on a color mask image.
		"""
		h, w, c = mask.shape

		plot_y = np.linspace(0, h - 1, h)
		coeffs = self.average_fit if average else self.last_fit_pixel

		line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]
		line_left_side = line_center - line_width // 2
		line_right_side = line_center + line_width // 2

		# Some magic here to recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array(list(zip(line_left_side, plot_y)))
		pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
		pts = np.vstack([pts_left, pts_right])

		# Draw the lane onto the warped blank image
		return cv2.fillPoly(mask, [np.int32(pts)], color)

	@property
	# average of polynomial coefficients of the last N iterations
	def average_fit(self):
		return np.mean(self.recent_fits_pixel, axis=0)

	@property
	# radius of curvature of the line (averaged)
	def curvature(self):
		y_eval = 0
		coeffs = self.average_fit
		return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

	@property
	# radius of curvature of the line (averaged)
	def curvature_meter(self):
		y_eval = 0
		coeffs = np.mean(self.recent_fits_meter, axis=0)
		return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])


def get_fits_by_sliding_windows(birdeye_binary, line_lt, line_rt, n_windows=9, verbose=False):
	"""
	Get polynomial coefficients for lane-lines detected in an binary image.

	:param birdeye_binary: input bird's eye view binary image
	:param line_lt: left lane-line previously detected
	:param line_rt: left lane-line previously detected
	:param n_windows: number of sliding windows used to search for the lines
	:param verbose: if True, display intermediate output
	:return: updated lane lines and output image
	"""
	height, width = birdeye_binary.shape

	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(birdeye_binary[int(height//1.5):-30, :], axis=0)

	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255

	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = len(histogram) // 2
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Set height of windows
	window_height = np.int(height / n_windows)

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = birdeye_binary.nonzero()
	nonzero_y = np.array(nonzero[0])
	nonzero_x = np.array(nonzero[1])

	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base

	margin = 100  # width of the windows +/- margin
	minpix = 50   # minimum number of pixels found to recenter window

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(n_windows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = height - (window + 1) * window_height
		win_y_high = height - window * window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Draw the windows on the visualization image
		cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
		cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low)
						  & (nonzero_x < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
						   & (nonzero_x < win_xright_high)).nonzero()[0]

		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
	line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

	detected = True
	if not list(line_lt.all_x) or not list(line_lt.all_y):
		left_fit_pixel = line_lt.last_fit_pixel
		left_fit_meter = line_lt.last_fit_meter
		detected = False
	else:
		left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
		left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

	if not list(line_rt.all_x) or not list(line_rt.all_y):
		right_fit_pixel = line_rt.last_fit_pixel
		right_fit_meter = line_rt.last_fit_meter
		detected = False
	else:
		right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
		right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)

	line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
	line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)

	# Generate x and y values for plotting
	ploty = np.linspace(0, height - 1, height)
	left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
	right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

	out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
	out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

	line_lt._x_coordinate = nonzero_x[left_lane_inds]
	line_lt._y_coordinate = nonzero_y[left_lane_inds]

	line_rt._x_coordinate = nonzero_x[right_lane_inds]
	line_rt._y_coordinate = nonzero_y[right_lane_inds]

	if verbose:
		f, ax = plt.subplots(1, 2)
		f.set_facecolor('white')
		ax[0].imshow(birdeye_binary, cmap='gray')
		ax[1].imshow(out_img)
		ax[1].plot(left_fitx, ploty, color='yellow')
		ax[1].plot(right_fitx, ploty, color='yellow')
		ax[1].set_xlim(0, 1280)
		ax[1].set_ylim(720, 0)

		plt.show()

	return line_lt, line_rt, out_img

def compute_offset_from_center(line_lt, line_rt, frame_width):
	"""
	Compute offset from center of the inferred lane.
	The offset from the lane center can be computed under the hypothesis that the camera is fixed
	and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation
	from the lane center as the distance between the center of the image and the midpoint at the bottom
	of the image of the two lane-lines detected.

	:param line_lt: detected left lane-line
	:param line_rt: detected right lane-line
	:param frame_width: width of the undistorted frame
	:return: inferred offset
	"""
	if line_lt.detected and line_rt.detected:
		line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
		line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
		lane_width = line_rt_bottom - line_lt_bottom
		midpoint = frame_width / 2
		offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
		offset_meter = xm_per_pix * offset_pix
	else:
		offset_meter = -1

	return offset_meter

#
# Just as Basic Function
def car_stairing_movement(line_lt, line_rt, offset):

	#try:
	#	left = np.polyfit(line_lt._x_coordinate, line_lt._y_coordinate, 3)
	#	right = np.polyfit(line_rt._x_coordinate, line_rt._y_coordinate, 3)
	#	wheel = left[0]*right[0]
	#except:
	#	wheel = 0
	if line_lt.detected and line_rt.detected:
		lx = line_lt._x_coordinate[0]-line_lt._x_coordinate[-1]
		rx = line_rt._x_coordinate[0]-line_rt._x_coordinate[-1]
		wheel = (lx+rx)/2.0 
	else:
		wheel = 0
	pid.setError(wheel*offset)
	pid.integral_error = offset
	return pid.totalerror()


def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter, stairing_movement):
    """
    Prepare the final pretty pretty output blend, given all intermediate pipeline images

    :param blend_on_road: color image of lane blend onto the road
    :param img_binary: thresholded binary image
    :param img_birdeye: bird's eye view of the thresholded binary image
    :param img_fit: bird's eye view with detected lane-lines highlighted
    :param line_lt: detected left lane-line
    :param line_rt: detected right lane-line
    :param offset_meter: offset from the center of the lane
    :return: pretty blend with all images and stuff stitched
    """
    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.3
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y+(h/4)), color=(10, 10, 10), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.4, src2=blend_on_road, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit
    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    if stairing_movement<0:
    	action = 'Turn Left {} deg'
    elif stairing_movement>0:
    	action = 'Turn Right {} deg'
    else:
    	action = 'Straight'
    cv2.putText(blend_on_road, 'Curvature radius   : {:.02f}m'.format(mean_curvature_meter), (w/3, (h/3)+20), font, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center : {:.02f}m'.format(offset_meter), (w/3, (h/3)+50), font, 0.9, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, action.format(stairing_movement), (w/3, (h/3)+80), font, 0.9, (0,255,255), 2, cv2.LINE_AA)
    return blend_on_road


def get_fits_by_previous_fits(birdeye_binary, line_lt, line_rt, verbose=False):
	"""
	Get polynomial coefficients for lane-lines detected in an binary image.
	This function starts from previously detected lane-lines to speed-up the search of lane-lines in the current frame.

	:param birdeye_binary: input bird's eye view binary image
	:param line_lt: left lane-line previously detected
	:param line_rt: left lane-line previously detected
	:param verbose: if True, display intermediate output
	:return: updated lane lines and output image
	"""

	height, width = birdeye_binary.shape

	left_fit_pixel = line_lt.last_fit_pixel
	right_fit_pixel = line_rt.last_fit_pixel

	nonzero = birdeye_binary.nonzero()
	nonzero_y = np.array(nonzero[0])
	nonzero_x = np.array(nonzero[1])
	margin = 100
	left_lane_inds = (
	(nonzero_x > (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] - margin)) & (
	nonzero_x < (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] + margin)))
	right_lane_inds = (
	(nonzero_x > (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] - margin)) & (
	nonzero_x < (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] + margin)))

	# Extract left and right line pixel positions
	line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
	line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

	detected = True
	if not list(line_lt.all_x) or not list(line_lt.all_y):
		left_fit_pixel = line_lt.last_fit_pixel
		left_fit_meter = line_lt.last_fit_meter
		detected = False
	else:
		left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
		left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

	if not list(line_rt.all_x) or not list(line_rt.all_y):
		right_fit_pixel = line_rt.last_fit_pixel
		right_fit_meter = line_rt.last_fit_meter
		detected = False
	else:
		right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
		right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)

	line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
	line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)

	# Generate x and y values for plotting
	ploty = np.linspace(0, height - 1, height)
	left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
	right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

	# Create an image to draw on and an image to show the selection window
	img_fit = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255
	window_img = np.zeros_like(img_fit)

	# Color in left and right line pixels
	img_fit[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
	img_fit[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
	#result = cv2.addWeighted(img_fit, 1, window_img, 0.3, 0)
	result = img_fit
	if verbose:
		plt.imshow(result)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)

		plt.show()

	return line_lt, line_rt, img_fit


def draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state):
	"""
	Draw both the drivable lane area and the detected lane-lines onto the original (undistorted) frame.
	:param img_undistorted: original undistorted color frame
	:param Minv: (inverse) perspective transform matrix used to re-project on original frame
	:param line_lt: left lane-line previously detected
	:param line_rt: right lane-line previously detected
	:param keep_state: if True, line state is maintained
	:return: color blend
	"""
	height, width, _ = img_undistorted.shape

	left_fit = line_lt.average_fit if keep_state else line_lt.last_fit_pixel
	right_fit = line_rt.average_fit if keep_state else line_rt.last_fit_pixel

	# Generate x and y values for plotting
	ploty = np.linspace(0, height - 1, height)
	left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
	right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

	# draw road as green polygon on original frame
	road_warp = np.zeros_like(img_undistorted, dtype=np.uint8)
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
	road_dewarped = cv2.warpPerspective(road_warp, Minv, (width, height))  # Warp back to original image space

	blend_onto_road = cv2.addWeighted(img_undistorted, 1., road_dewarped, 0.3, 0)

	# now separately draw solid lines to highlight them
	line_warp = np.zeros_like(img_undistorted)
	line_warp = line_lt.draw(line_warp, color=(255, 0, 0), average=keep_state)
	line_warp = line_rt.draw(line_warp, color=(0, 0, 255), average=keep_state)
	line_dewarped = cv2.warpPerspective(line_warp, Minv, (width, height))

	lines_mask = blend_onto_road.copy()
	idx = np.any([line_dewarped != 0][0], axis=2)
	lines_mask[idx] = line_dewarped[idx]

	blend_onto_road = cv2.addWeighted(src1=lines_mask, alpha=0.8, src2=blend_onto_road, beta=0.5, gamma=0.)

	return blend_onto_road



def main():
	global line_lt, line_rt, processed_frames
	processed_frames = 0                    # counter of frames processed (when processing video)
	line_lt = Line(buffer_len=10)  # line on the left of the lane
	line_rt = Line(buffer_len=10)  # line on the right of the lane
	im = np.load('./../testing_data/img.npy')
	line_lt, line_rt, img_fit = get_fits_by_sliding_windows(im, line_lt, line_rt, n_windows=11, verbose=True)
	
	return
	

if __name__ == '__main__':
	main()
