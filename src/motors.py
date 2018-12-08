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



import RPi.GPIO as GPIO #Import GPIO library
import time


# Configurations

class HardwareApi:
	def __init__(self):
		# Set UP GPIO to BCM
		GPIO.setmode(GPIO.BCM)
		
	def __enter__(self):
		return self

	def close(self):
		GPIO.cleanup()
		return

	def __exit__(self, *args, **kwargs):
		return self.close()

	  
	  
	  
class DriveTurnings:
	def __init__(self, ports=[15, 23]):
		self.motion = 0
		self.ports = ports
		self.initialise()

	def initialise(self):
		for i in self.ports:
			GPIO.setup(i, GPIO.OUT)
		return


	def move(self, port):
		print '[+] Port Starting..', port
		# Stop Preview Motion PORT
		if self.motion:
			GPIO.output(self.motion, GPIO.LOW)
		GPIO.output(port, GPIO.HIGH)
		self.motion = port
		return

	def stop(self):
		GPIO.output(self.motion, GPIO.LOW)
		self.motion = 0
		return

	def moveRight(self):
		self.move(self.ports[0])
		return

	def moveLeft(self):
		self.move(self.ports[1])
		return

		
	
class DriveAccleration:
	def __init__(self, ports=[18,14]):
		# Pin Connections
		# 18 : Forward
		# 14 : Backward
		# 15 : Right
		# 23 : Left
		#
		# Set UP GPIO to BCM
		#GPIO.setmode(GPIO.BCM)
		self.motion = 0
		self.ports = ports
		self.initialise()


	def initialise(self):
		for i in self.ports:
			GPIO.setup(i, GPIO.OUT)
		return


	def move(self, port):
		print '[+] Port Starting..', port
		# Stop Preview Motion PORT
		if self.motion:
			GPIO.output(self.motion, GPIO.LOW)
		GPIO.output(port, GPIO.HIGH)
		self.motion = port
		return

	def stop(self):
		GPIO.output(self.motion, GPIO.LOW)
		self.motion = 0
		return

	def moveForward(self):
		self.move(self.ports[0])
		return

	def moveBackward(self):
		self.move(self.ports[1])
		return

def main():
	with HardwareApi() as gpio_handler:
		acc = DriveAccleration()
		mov = DriveTurnings()
		
		print '[*] Starting Testing.'
		print '[*] Turning Stairing Left'
		mov.moveLeft()
		time.sleep(3)
		print '[+] Running Backward'
		acc.moveBackward()
		time.sleep(3)
		print '[+] Running Forward'
		acc.moveForward()
		time.sleep(3)
		print '[+] Running Left'
		acc.stop()
		time.sleep(3)
		print '[+] Running Right'
		mov.moveRight()
		time.sleep(3)
		mov.stop()
		return



if __name__ == '__main__':
	main()