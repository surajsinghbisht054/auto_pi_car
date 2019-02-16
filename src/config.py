
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



# import moduels
import math


# select True, if you want to use Desktop Or Laptop Settings
# Or Select False, For Rasberry Pi Settings.
comp = True
print '[*] Using Computer Setttings : ', comp

# name of window previewing window
WINDOW_PREVIEWER = 'Receiving Image'

# set debug to True
DEBUG = True

# Select To Use GTK Graphic Lib
GTK = True #  Select False, To Use Matplotlib (jupyter Notebook)
print '[*] Use GTK Graphic Lib : ', GTK

# Default Operational Image Height And Width
WIDTH = 1024
HEIGHT = 600

# Camera Resolution
(RX, RY) = (640, 480)
print '[*] Camera Resolution : ', (RX, RY)

# OpenCv Resize 
(RW, RH) = (125, 44)
print '[*] OpenCV Resizing : ', (RW, RH)


# Focus On Road Region Of Interest
(cx1, cy1, cx2, cy2) = (0,int(HEIGHT/2.3), WIDTH, HEIGHT)

#



# Training Screen Focus Region Highlight 
(X,W,Y,H) = (0,RX, 0+300, RY-300)





# STREAM VIDEO Input IP
STREAM_VIDEO_INPUT_IP = '127.0.0.1' 
STREAM_VIDEO_INPUT_IP = '192.168.43.116'
print '[*] Stream Input listening IP address : ', STREAM_VIDEO_INPUT_IP

# STREAM VIDEO INPUT PORT
STREAM_VIDEO_INPUT_PORT = 8002
print '[*] Stream Input listening Port address : ', STREAM_VIDEO_INPUT_PORT
