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
#
#

__author__ 	    = "Suraj Singh Bisht"
__description__ = "Self Driving Phase One"
__version__	    = "Beta"
__date__	    = "OCT, 2018"
__email__	    = "surajsinghbisht054@gmail.com"


import cv2
from utili import imshow
import time
import numpy as np


class RedLightDetection:
	def __init__(self, arrayrange):
		self.arrayrange = arrayrange

	def detect(self, iframe):
		frame = cv2.medianBlur(iframe, 5)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		img = np.zeros(shape=frame.shape[:2], dtype=np.uint8)
		for l, u in self.arrayrange:
			mask = cv2.inRange(frame, l, u)
			img = cv2.bitwise_or(img, mask)
			img = cv2.bitwise_and(iframe, iframe, mask=img)
		return img


def main():
	path = '../temporary/redlight.mp4'
	cam = cv2.VideoCapture(path)
	d=RedLightDetection([(np.array([121,129,123]), np.array([255, 200, 255]))])
	while cam.isOpened():

		ret, frame = cam.read()
		if not ret:
			break
		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		frame = cv2.resize(frame, (1080,720))

		if imshow('GaussianBlur', d.detect(frame)):
			break
		if imshow('Preview', frame):
			break
		time.sleep(0.01)
	return

if __name__ == '__main__':
	main()