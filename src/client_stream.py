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
import time
import struct
from config import *
import io 
import numpy as np 

# Sent Image Continously
class ImageRail:
	def __init__(self, sock):
		self.sock = sock 
		self.mstream = self.sock.makefile(mode='wb') # Make File Handler
		self.fstream = io.BytesIO() # In-Memory Byte Storing Class
		self.initialise_setups()


	def initialise_setups(self):
		print '[*] Going To Start Streaming Live Images..'
		with pi_api.PiCamera() as cam:
			while True: # Use While For Continue Streaming
				self.fstream.seek(0)
				#print '[*] First Image Streamed.'
				img = cam.getimage()
				np.save(self.fstream, img)
				#self.fstream.seek(0)
				imgsize = self.fstream.tell() # image Size
				self.fstream.seek(0)
				self.mstream.write(struct.pack('I', imgsize))
				#print imgsize
				self.mstream.flush()
				#time.sleep(1)
				self.mstream.write(self.fstream.read(imgsize))
				self.mstream.flush()
				#print '[*] Image Sent'
		return




class Streamer:
	def __init__(self, ip, port, listen):
		self.ip = ip
		self.port = port
		self.listen = listen
		self.initialize_setups()

	def __enter__(self):
		return self

	def __exit__(self, *args, **kwargs):
		return self.close()


	def start(self):
		# Use While 
		if True:
			# Receive Connection
			print '[*] Waiting For Connection......'
			(clientsocket, address) = self.sock.accept()
			# Not Using Any Multi-Thread, Because Only One Core System Needed..
			print '[*] Receive Connection : ', address
			# Image Rail Activated
			ImageRail(clientsocket)

		return

	def close(self):
		print '[*] Socket Close'
		self.sock.close()
		return

	def initialize_setups(self):
		# create Socket
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		# bind socket
		self.sock.bind((self.ip, self.port))
		# listen
		self.sock.listen(self.listen)

		try:
			# Start Server
			self.start()
		except Exception as E:
			print '[-] An Error Caught :- ', E
			self.close()
		return


# main function
def main():
	with Streamer(STREAM_VIDEO_INPUT_IP, STREAM_VIDEO_INPUT_PORT, 5) as obj:
		pass
	return


# main trigger function
if __name__=='__main__':
	main()
