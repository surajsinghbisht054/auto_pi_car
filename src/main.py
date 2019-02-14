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


from server_stream import DStream
from config import WINDOW_PREVIEWER
from machine_view import DetectTrack
from utili import FocusRegion
from utili import imshow


# For Server
def desktop_main():

	# Automatic Content Closing Method
	with DStream() as obj:
		# machine view
		mview = DetectTrack()

		# Continouse Image Receiving
		while True:
			try:
				# Image Received 
				# Image in Original Form
				im  = obj.getimage()

				# Image Without Any Processing
				mview.setimg(im)
				# Adjustment For Rasbi Pi Car
				adjustment =  [[400,300],[1280-350,300]]
				#adjustment = False
				im = mview.highlight(adjustment, sock=obj.sock)
				# Image Without Any Processing
				if imshow('Machine View', im):
					break

			
			except Exception as e:
				print e
				break

	return



