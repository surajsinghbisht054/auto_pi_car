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
import time
import matplotlib.pyplot as plt
import numpy as np
from config import *	
import math 
from utili import get_fits_by_sliding_windows, Line, draw_back_onto_the_road, imshow, get_fits_by_previous_fits, compute_offset_from_center, prepare_out_blend_frame, car_stairing_movement
from adjustments_window import AdjustmentWindow
from struct import pack


#
# Range of Yellow Color Pixel To Detect All Yellow Pixels From Image
# (White Pixel Range Built in functionality)
#
yellow_HSV_th_min = np.array([0, 70, 70])
yellow_HSV_th_max = np.array([50, 255, 255])
processed_frames = 0                    # counter of frames processed (when processing video)
line_lt = Line(buffer_len=10)  # line on the left of the lane
line_rt = Line(buffer_len=10)  # line on the right of the lane
pas = 0 


# Filter Only Required Pixel Range From Image Array
def thresh_frame_HSV(frame, minthrash, maxthrash, verbose=False):
	# convert to HSV
	tmp_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
	"""
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	eq_global = cv2.equalizeHist(gray)

	_, th = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)
	
	return th

	

# Sobel Thresh
def thresh_frame_sobel(frame, kernel_size):
	"""
	Apply Sobel edge detection to an input frame, then threshold the result
	"""
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
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
# Same Procedure With Sobel & morphology.
#
# 
def highlight_tracks(im, verbose=False):
	'''
	highlight_tracks
		function to perform all lane highlighting call function in sequence
	'''

	# create blank copy
	binary_image_array = np.zeros(shape=im.shape[:2], dtype=np.uint8)

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
		
	# Another Technique
	c= thresh_frame_sobel(im, 9)
	
	# Blend Again
	binary_image_array = np.logical_or(binary_image_array, c)
	
	if verbose:
		plt.imshow(c)
		plt.show()
		
	# more detail highligting
	kernel = np.ones((5, 5), np.uint8)
	
	return cv2.morphologyEx(binary_image_array.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
	#return binary_image_array

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
def MachineView(frame, verbose= False, keep_state=True,V1=[350, 560], V2=[1280-350, 560], socket=None):
	global line_lt, line_rt, processed_frames

	
	if verbose:
		plt.imshow(frame)
		plt.show()
	
	# Perform Highlighting Functions
	im = highlight_tracks(frame, verbose=verbose)
	
	if verbose:
		plt.imshow(im)
		plt.show()

	# Perform Eagle View
	tm, revtm = EagleView(im, tl=V1, tr=V2) # <--- Adjust This As Your Requirements
	
	if verbose:
		plt.imshow(tm)
		plt.show()

	# Get Means Result of Traces
	#p1,p2,p3,p4 = get_mean_x(tm)


	if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
		line_lt, line_rt, img_fit = get_fits_by_previous_fits(tm, line_lt, line_rt, verbose=False)
	else:
		line_lt, line_rt, img_fit = get_fits_by_sliding_windows(tm, line_lt, line_rt, n_windows=9, verbose=False)

	#line_lt, line_rt, img_fit = get_fits_by_sliding_windows(tm, line_lt, line_rt, n_windows=6, verbose=False)

	blend_on_road = draw_back_onto_the_road(frame, revtm, line_lt, line_rt, True)
	processed_frames += 1
	offset_meter = compute_offset_from_center(line_lt, line_rt, tm.shape[1])
	movement = car_stairing_movement(line_lt, line_rt, tm.shape[1])
	blend_on_road = prepare_out_blend_frame(blend_on_road, frame, tm, img_fit, line_lt, line_rt, offset_meter, movement)
	taketurn = 0
	if 300>movement>200:
		taketurn = 0
	elif 200>movement:
		taketurn = 1
	elif movement>700:
		taketurn = 1
	else:
		taketurn = 2

	if (processed_frames<20):
		taketurn = 0

	# first for movement, second for turns
	if socket:
		socket.send(pack('bb', 0, taketurn))
	if processed_frames==20:
		processed_frames=-1
	# Assembly points as numpy array
	#pts = np.array([[p1,0],[p2,0],[p4,700],[p3,700]], np.int32)
	
	# create blank image to plot mean result points
	#tm = np.zeros(shape=tm.shape)
	#print tm.shape
	#tm = np.zeros((tm.shape[0],tm.shape[1],3), np.uint8)
	
	# plot all mean result points as Poly Shape And Fill White Color
	#cv2.fillPoly(tm,[pts],(255,255,255))
	
	#if verbose:

	#	plt.imshow(tm, cmap='gray')
	#	plt.show()

	# Now, Time To Perform Reverse Eagle View Calculations
	#tm = EagleView(tm, tl=[V1, V2], tr=[1280-V1, V2], rev=True)
	#tm = cv2.warpPerspective(tm, revtm, (frame.shape[1], frame.shape[0]))
	
	#if verbose:
	#	plt.imshow(tm, cmap='gray')
	#	plt.show()

	#print tm
	#tm = cv2.cvtColor(tm, cv2.COLOR_BGR2RGB)

	#dst = cv2.addWeighted(frame,0.5, tm, 0.5, 0)
	#plt.imshow(blend_on_road)
	#plt.show()
	return blend_on_road




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


# class for lan detection
class DetectObject:
	def __init__(self):
		'''
		Here, We can Pass Various Options And Arguments 
		For Default Purposes.
		
		'''
		self.img = None
		self.coor = []
		pass
	
	def load(self, imgpath):
		self.img = cv2.imread(imgpath)
		return
	
	def getimg(self, img):
		self.img = img
		return
	
	def highlight(self, adjustment, sock=None):
		global pas
		if adjustment:
			v1, v2 = adjustment
			img = MachineView(self.img, verbose= False, V1=v1, V2=v2, socket=sock)
		else:
			img = True
			if sock:
				sock.send(pack('bb', 0, 0))
			if pas>10:
				root = AdjustmentWindow(self.img, className=' Quick Adjustment coordinates Finder')
				root.mainloop()
		pas += 1
		return img


	
		# highlight Lines 
	def create_lines(self, img, lis):
		for coor in lis:
			for x1,y1,x2,y2 in coor:
				if abs(y2-y1)<40:
					continue
				img = cv2.line(img, (x1,y1), (x2,y2), (255,127,200), 5)
		return img
	

	
		# Performing Edge Detection Function Here
	def perform_edge(self, img):
		return cv2.Canny(img, threshold1= 10, threshold2=50)
	
	
		# performing Gaussian Blur Function Here
	def perform_blur(self, img):
		return cv2.GaussianBlur(img, (5, 5), 0)
	
		# Resizing Function
	def perform_resize(self, img):
		return cv2.resize(img, (WIDTH, HEIGHT))
	
		# perform gray scale conversion
	def perform_gray_conv(self, img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	


	

# Main Function 
def main():
	path = '../testingData/test2.jpg'
	Dobj = DetectObject()
	Dobj.load(path)
	adjustment = [[570,450],[710,450]]
	frame = Dobj.highlight(adjustment)

	plt.imshow(Dobj.img)
	plt.show()
	return

def second_main():
	path = '../testingData/deskvideo.mp4'
	Dobj = DetectObject()
	video = cv2.VideoCapture(path)
	while video.isOpened():
	#for i in range(150):
		ret, frame = video.read()
		if ret:
			Dobj.img = frame
			adjustment = [[570,450],[710,450]]
			frame = Dobj.highlight(adjustment)
			if imshow('PreView', frame):
				break
		else:
			break

	return


if __name__ == '__main__':
	main()
	second_main()