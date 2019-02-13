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


# import modules
import socket
import pi_api
import cv2
import time
import struct
from config import *
import io 
import numpy as np 
from utili import imshow
from PIL import Image
import zlib


# Server Side Class To Receive Image As np array From Network
class DStream:
	'''
	DStream :
		function to receive live image data from connect socket and 
			convert it to cv2.imread object.

	'''
	# initialise function
	def __init__(self):
		self.sock = socket.socket()
		self.connection()
		self.createFileBuffers()

	# file buffer handling variables
	def createFileBuffers(self):	
		self.fstream = io.BytesIO()
		self.mstream = self.sock.makefile('rb')
		return

	# get image from socket
	def getimage(self):
		self.fstream.seek(0)
		imgsize = struct.unpack('<L', self.mstream.read(4))[0]
		self.fstream.write(zlib.decompress(self.mstream.read(imgsize)))
		return cv2.cvtColor(np.asarray(Image.open(self.fstream)), cv2.COLOR_RGB2BGR)


	# establish socket connection
	def connection(self):
		while True:
			try:
				print '[*] Trying To Connect...'
				self.sock.connect((STREAM_VIDEO_INPUT_IP, STREAM_VIDEO_INPUT_PORT))
				break
			except socket.error, E:
				print '	[-] Socket Not Connecting. ', E
				print '	[-] Trying IP : {} PORt : {}'.format(STREAM_VIDEO_INPUT_IP, STREAM_VIDEO_INPUT_PORT)
				time.sleep(3)
		return

	def __enter__(self, *args, **kwargs):
		return self

	def close(self):
		return self.sock.close()

	def __exit__(self, *args, **kwargs):
		return self.close()



# main function
def main():
	with DStream() as obj:
		while True:
			try:
				NAME = 'PREVIEW'
				img = obj.getimage()
				if imshow(NAME, img):
					break
			except Exception as e:
				print e

	return

if __name__ == '__main__':
	main()
