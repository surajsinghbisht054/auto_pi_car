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



class PID:
	def __init__(self, proportional, integral, derivative):
		self.proportional = proportional
		self.integral = integral
		self.derivative = derivative
		self.proportional_error = 0.0 
		self.integral_error = 0.0
		self.derivative_error = 0.0


	def setError(self, error):
		error = float(error)
		self.integral_error += error # past
		self.proportional_error = error # present
		self.derivative_error = self.proportional_error - error # trying to predict future
		return

	def totalerror(self):
		return -((self.proportional * self.proportional_error)+(self.integral * self.integral_error)+(self.derivative * self.derivative_error))


def main():
	pid = PID(0.1, 0.0001, 1.0)
	for i in [1,2,3,4,-1,-2,-3,0,0,1.2,1.5]:
		pid.setError(i)
		print "[*] Error : {} | Action : {}".format(i, pid.totalerror())
	return

if __name__ == '__main__':
	main()