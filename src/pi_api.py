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
__description__ = "Auto-Pilot Rasberry Pi Car Project"
__version__	    = "Beta"
__date__	    = "SEPT, 2018"
__email__	    = "surajsinghbisht054@gmail.com"


# importing configuration file
from config import comp, RX, RY

# if rasberry pi
if not comp:
	from picamera import PiCamera as pcam
	from picamera.array import PiRGBArray

# In Computer
else:
	import cv2

# import required modules
import time




# creating camera handling Api
class PiCamera:
	'''
		Desktop And Rasberry Pi, Cross Hardware Platform Supported Class Object.
		It Automatically Handles All Internal Changes After Verify Config File Setting
		'Comp' value.

		Functionality Supported By PiCamera
		
			rotation(rotation) 
			setResolution(resolution)
			capture(filename=str(time.time())+'.jpg')
			close()
			release()
			read()
			getimage()



	'''
	def __init__(self, resolution=(RX, RY), rotation=0):
		'''
		Initialise Class Object
		'''
		# if rasberry pi:
		if not comp:
			# Initialise PiCamera Object
			self.cam = pcam()
			
		# if Computer
		else:
			self.cam = cv2.VideoCapture(0)

		# set resolution
		self.setResolution(resolution)

		
		# rotation
		self.rotation(rotation)
		
		# if rasberry pi
		if not comp:
			# Initialise Image Array
			self.array = PiRGBArray(self.cam)
		else:
			self.array = []
		pass

	def __enter__(self, *args, **kwargs):
		return self



	def rotation(self, rotation):
		if not comp:
			self.cam.rotation = rotation
		return


	# Set Camera resolution
	def setResolution(self, resolution):

		# if computer
		if comp:
			self.cam.set(3, resolution[0])
			self.cam.set(4, resolution[1])

		else:
			# Set Resolution
			self.cam.resolution = resolution
		return

	# Capture Image
	def capture(self, filename='../temporary/'+str(time.time())+'.jpg'):
		# Camera Ready
		time.sleep(0.2)
		self.cam.capture(filename)
		return

	def close(self):
		return self.release()

	def __exit__(self, *args, **kwargs):
		return self.release()

	# close Cam
	def release(self):
		print '[*] Releasing Camera'
		# rasberry pi
		if not comp:
			self.cam.close()
		else:
			self.cam.release()
		return

	def read(self):
		# rasberry pi
		if not comp:
			return (None, self.getimage())
		else:
			return self.cam.read()

	# Get Image
	def getimage(self):
		# rasberry pi
		if not comp:
			self.array.truncate(0)
			self.cam.capture(self.array, format='bgr')
			# Returning Numpy Array
			return self.array.array
		# Computer
		else:
			return self.read()[1]



# Main Function
def main():
	try:
		cam = PiCamera()
		# if rasberry py
		if not comp:
			cam.capture()

		cv2.imwrite('../temporary/save.jpg', cam.getimage())


	except Exception as e:
		print e
	finally:
		cam.release()
	return



# Main Function Trigger
if __name__=='__main__':
	main()
