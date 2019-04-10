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
#           Please Don't Remove Author Initials.
#			Inspired By Udacity Self Driving Car Enginner Nanodegree.
#

__author__      = "Suraj Singh Bisht"
__description__ = "Self Driving Phase One"
__version__     = "Beta"
__date__        = "OCT, 2018"
__email__       = "surajsinghbisht054@gmail.com"


# Script for Image LIne Detection

# importing modules
import cv2
import os
import numpy as np
import math 
import time
import matplotlib.pyplot as plt
from struct import pack

# framework modules
from config import *	
from utili import get_fits_by_sliding_windows,Line, draw_back_onto_the_road, imshow, get_fits_by_previous_fits, compute_offset_from_center, prepare_out_blend_frame, car_stairing_movement
from adjustments_window import AdjustmentWindow

if __name__ == '__main__':
	ACCURATE = True
	SOBEL = False
else:
	ACCURATE = False
	SOBEL = False


LANE_REFRESH_ITERATION = 25

#
# Range of Yellow Color Pixel To Detect All Yellow Pixels From Image
# (White Pixel Range Built in functionality)
#
yellow_HSV_th_min = np.array([0, 70, 70])
yellow_HSV_th_max = np.array([50, 255, 255])
processed_frames = 0                    # counter of frames processed (when processing video)
line_lt = Line(buffer_len=12)  # line on the left of the lane
line_rt = Line(buffer_len=12)  # line on the right of the lane
pas = 0 




# Filter Only Required Pixel Range From Image Array
def thresh_frame_HSV(frame, minthrash, maxthrash, verbose=False):
	'''
	Filter Pixel Range From Image.

		frame :
				ImageArray object
		minthrash :
				Minimum Thrash Array Range
		maxthrash :
				Maximum Thrash Array Range

	'''

	# convert to HSV
	tmp_image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)

	# Filter Pixel Range
	otmp_image = cv2.inRange(tmp_image, minthrash, maxthrash)

	if verbose:
		plt.imshow(otmp_image, cmap='gray')
		plt.show()
	return otmp_image



# filter white pixel, completely like above function
def get_binary_image_array_from_equalized_grayscale(frame):
	"""
	Apply histogram equalization to an input frame, threshold it and return the (binary_image_array) result.

	frame :
			ImageArray object

	"""
	
	eq_global = cv2.equalizeHist(frame.gray)
	#_, th = cv2.threshold(eq_global, thresh=240, maxval=255, type=cv2.THRESH_BINARY)
	_, th = cv2.threshold(eq_global, thresh=240, maxval=255, type=cv2.THRESH_TOZERO)
	return th


# canny edge detection
def canny_edge_detection(frame):
	'''
	frame :
			ImageArray object

	'''	
	return cv2.Canny(frame.image, threshold1= 60, threshold2=250)

# Sobel Thresh
def thresh_frame_sobel(frame, kernel_size):
	"""
	Apply Sobel edge detection to an input frame, then threshold the result
	"""
	gray = frame.gray
	# side x
	sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
	# side y
	sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

	sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
	sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

	_, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)
	return sobel_mag.astype(bool)


#
#
# This Function, Call All Details Highlighting Function
# One By one in Sequence. hence, You Doesn't Need To Worry
# About Anything. Just Pass Your Image As im parameter.
# 
# Thresh frame HSV (Filter Pixel Range From Image)
# Then, Paste Pixel Details Into Blank Image
# Get All White Pixel, After Performing Gray Scale
# Then, Paste Result Details In Previous Used Blank Image
# Same Procedure With Sobel/CannyEdge & morphology.
#
# 
def highlight_tracks(im, verbose=False):
	'''
	highlight_tracks
		function to perform all lane highlighting call function in sequence
	'''

	# create blank copy
	binary_image_array = np.zeros(shape=im.image.shape[:2], dtype=np.uint8)

	# HSV Yellow Color
	c=thresh_frame_HSV(im, yellow_HSV_th_min, yellow_HSV_th_max)

	# Blend With Copy
	binary_image_array = np.logical_or(binary_image_array, c)

	if verbose:
		plt.imshow(c)
		plt.show()

	# White Equalized
	c = get_binary_image_array_from_equalized_grayscale(im)
	
	# Again Blend
	binary_image_array = np.logical_or(binary_image_array, c)
	
	if verbose:
		plt.imshow(c)
		plt.show()
		
	# Sobel / Canny Edge Detection
	if SOBEL:
		# Another Technique
		c= thresh_frame_sobel(im, 9)
	else:
		c = canny_edge_detection(im)

	# Blend Again
	binary_image_array = np.logical_or(binary_image_array, c)
	
	if verbose:
		plt.imshow(c)
		plt.show()
		
	# more detail highligting
	kernel = np.ones((5, 5), np.uint8)
	return cv2.morphologyEx(binary_image_array.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
	

# To Perform Bird View Transformation
def EagleView(tmp_image, tl=[], tr=[], dif = 10):
	'''
	Eagle View Function To Perform Bird Eye View Transformations In Easiest Way

		Param
			tmp_image = Input Image
			tl = [left point coordinate]
			tr = [right point coordinate]
			dif = [little bit adjustment in calculation]

	'''
	# get image shape
	(H, W) = tmp_image.shape[:2]
	
	# input coordinates of image
	inp = [
		tl,
		tr,
		[0, H-10],
		[W, H-10]
	]
	# output coordinates of points
	out= [
		[0,0],
		[W, 0],
		[0, H],
		[W, H]
	]
	# reverse 
	rev_perp_data = cv2.getPerspectiveTransform(np.float32(out), np.float32(inp))
	# straigth
	perp_data = cv2.getPerspectiveTransform(np.float32(inp), np.float32(out))
	
	# perform transformations
	obj = cv2.warpPerspective(tmp_image, perp_data, (W, H))
	#objrev = cv2.warpPerspective(tmp_image, rev_perp_data, (W, H))
	return (obj, rev_perp_data)


#
# To Calculate mean of Available Road Pixel Traces [Testing Phase Function] (Not Using)
#
def get_mean_x(tm):
	np.save('img', tm)
	y,x = tm.shape
	cim = np.zeros(shape=tm.shape)

	lines = cv2.HoughLinesP(tm,cv2.HOUGH_PROBABILISTIC,np.pi/180,20,5, 20)
	for z in lines:
		for x1,y1,x2,y2 in z:
			cv2.line(cim, (x1,y1), (x2, y2), (255,255,255), 20)
	# Divide Into 4 Section
	pt1 = cim[0:y/2, 0:x/2] 
	pt2 = cim[0:y/2, x/2:x ]
	pt3 = cim[y/2:y, 0:x/2]
	pt4 = cim[y/2:y, x/2:x]
	
	return [np.mean(pt1.nonzero()[0]),
	np.mean(pt2.nonzero()[0])+int(x/2),
	np.mean(pt3.nonzero()[0]),
	np.mean(pt4.nonzero()[0])+int(x/2),]

# Composite Of All Above Function, To Detect Lane
def MachineView(frameObj, verbose= False, keep_state=True,V1=[350, 560], V2=[1280-350, 560]):
	global line_lt, line_rt, processed_frames

	
	if verbose:
		plt.imshow(frameObj.image)
		plt.show()
	
	tm, revtm = EagleView(frameObj.image, tl=V1, tr=V2) # <--- Adjust This As Your Requirements
	
	# Perform Highlighting Functions
	tm = highlight_tracks(ImageArray(tm), verbose=verbose)
	
	if verbose:
		plt.imshow(tm)
		plt.show()
	
	if verbose:
		plt.imshow(tm)
		plt.show()

	if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected and not ACCURATE:
		line_lt, line_rt, img_fit = get_fits_by_previous_fits(tm, line_lt, line_rt, verbose=False)
	else:
		line_lt, line_rt, img_fit = get_fits_by_sliding_windows(tm, line_lt, line_rt, n_windows=8, verbose=False)

	blend_on_road = draw_back_onto_the_road(frameObj.image, revtm, line_lt, line_rt, True)
	processed_frames += 1
	offset_meter = compute_offset_from_center(line_lt, line_rt, tm.shape[1])
	movement = car_stairing_movement(line_lt, line_rt, offset_meter)
	blend_on_road = prepare_out_blend_frame(blend_on_road, frameObj.image, tm, img_fit, line_lt, line_rt, offset_meter, movement)

	if processed_frames==LANE_REFRESH_ITERATION:
		processed_frames=-1
	#print "[*] Line X1 : {} | Y1 : {} | X-1 : {} | Y-1 : {} | Offset : {} | Movement : {}".format(line_rt.all_x[1],line_rt.all_y[1],line_rt.all_x[-1], line_lt.all_y[-1], offset_meter, movement)
	
	return (blend_on_road, offset_meter, movement)




# [Testing Phase Function] (Not Using)
def getMedianLines(coor):
	return



# Find Slope [Testing Phase Function] (Not Using)
def findslope(x1, y1, x2, y2):
	# Convert Integer Type To float 32
	x1 = np.float32(x1)
	y1 = np.float32(y1)
	x2 = np.float32(x2)
	y2 = np.float32(y2)
	slope = (y2 - y1) / (x2 - x1 + np.finfo(float).eps)
	bias = y1 - slope * x1
	return (slope, bias)

# class to store image data as object
class ImageArray:
	def __init__(self, im):
		self.image = im
		self.gray = self.gray_image()
	def gray_image(self):
		return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


# class for lan detection
class DetectTrack:
	def __init__(self):
		'''
		Here, We can Pass Various Options And Arguments 
		For Default Purposes.
		
		'''
		self.img = None
		self.imgarr = None
		self.coor = []
		pass
	
	def load(self, imgpath):
		self.img = cv2.imread(imgpath)
		self.imgarr = self.setimg(self.img)
		return
	
	def setimg(self, img):
		self.img = img
		self.imgarr = ImageArray(self.img)
		return

	def getimg(self):
		return self.img
	
	def highlight(self, adjustment):
		global pas

		if adjustment:
			v1, v2 = adjustment
			img, offset, movement = MachineView(self.imgarr, verbose= False, V1=v1, V2=v2)
		else:
			offset = movement = None
			img = True
			if pas>10:
				root = AdjustmentWindow(self.img, className=' Quick Adjustment coordinates Finder')
				root.mainloop()
			pas += 1
		return (img, offset, movement)



# Main Function 
def main():
	path = '../testing_data/test2.jpg'
	detectObj = DetectTrack()
	detectObj.load(path)
	adjustment = [[570,450],[710,450]]
	frame = detectObj.highlight(adjustment)

	plt.imshow(detectObj.img)
	plt.show()
	return

def second_main():
	# path of testing video
	if True:
		path = '../testing_data/deskvideo.mp4'
		adjustment = [[570,450],[710,450]]
	else:
		path = '../temporary/highway4.mp4'
		adjustment = [[300,340],[554,340]]
	
	
	# detect object class object
	detectObj = DetectTrack()
	
	# cam accessing object
	video = cv2.VideoCapture(path)
	
	# bird view adjustments
	#while video.isOpened():
	for i in range(150):
		
		# read image
		ret, frame = video.read()
		#frame = cv2.flip(frame, 1)
		
		if ret:
			# load image
			detectObj.setimg(frame)

			# perform track highligting algo functions
			frame, offset, movement = detectObj.highlight(adjustment)
			#print "[*] Offset : {} | movement : {}".format(offset, movement)
			
			# show image
			if imshow('Processed Video Frames', frame):
				break
			#time.sleep(4)
		else:
			break

	return


if __name__ == '__main__':
	#main()
	start = time.time()
	second_main()
	end = time.time()
	print "[*] Starting Time : {} | Ending time : {} | Time Consumed : {}".format(start, end, end-start)