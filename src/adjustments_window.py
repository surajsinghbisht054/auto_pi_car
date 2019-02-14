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



# import module
import cv2
import Tkinter
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import ttk
import numpy as np
from utili import FocusRegion




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



class AdjustmentWindow(Tkinter.Tk):
	'''
	Tkinter Window To Find/Adjust Image Focus Region.
		Just Open Window, Pass The Path of Image 
			And Find Adjusting Point

	'''
	# initializing function
	def __init__(self, frame, *args, **kwargs):

		Tkinter.Tk.__init__(self, *args, **kwargs)
		self.defaultimage= frame
		self.controls = []
		self.initialise_setup()


	# sequence calling function
	def initialise_setup(self):
		self.create_controls()
		self.canvas_obj = self.create_canvas()

		# Method To Change/Set Image
		self.tkimage(self.defaultimage)
		self.setimage()

		self.load_default_values()
		return

	# load default values in entry boxes
	def load_default_values(self):
		for i in self.controls:
			i.delete(0, 'end')

		self.controls[0].insert(0, 0) # x1
		self.controls[1].insert(0, 0) # y1
		self.controls[2].insert(0, self.canvas_obj.imgwidth) # x2
		self.controls[3].insert(0, self.canvas_obj.imgheight) # y2

		# X
		self.controls[4].insert(0, 1)
		self.controls[5].insert(0, self.canvas_obj.imgwidth)

		self.controls[6].insert(0, 1)
		self.controls[7].insert(0, 1)
		return

	# create controls
	def create_controls(self):
		mainframe = ttk.Frame(self)
		mainframe.pack(ipadx=10, ipady=10)

		frame = ttk.LabelFrame(mainframe, text='Focus Region')
		frame.pack(ipadx=10, ipady=10, side='left', padx=10, pady=10)

		# All labels
		l1 = Tkinter.Label(frame, text=' x1 ')
		l2 = Tkinter.Label(frame, text=' y1 ')
		l3 = Tkinter.Label(frame, text=' x2 ')
		l4 = Tkinter.Label(frame, text=' y2 ')
		l1.grid(column=1, row=1)
		l2.grid(column=1, row=2)
		l3.grid(column=1, row=3)
		l4.grid(column=1, row=4)

		# All Entries
		in1 = Tkinter.Entry(frame, width=10)
		in2 = Tkinter.Entry(frame, width=10)
		in3 = Tkinter.Entry(frame, width=10)
		in4 = Tkinter.Entry(frame, width=10)
		in1.grid(column=2, row=1)
		in2.grid(column=2, row=2)
		in3.grid(column=2, row=3)
		in4.grid(column=2, row=4)

		frame = ttk.LabelFrame(mainframe, text='Bird View Changes')
		frame.pack(ipadx=10, ipady=10, side='left', padx=10, pady=10)

		# All labels
		l1 = Tkinter.Label(frame, text=' P1(x,y) ')
		l2 = Tkinter.Label(frame, text=' P2(x,y) ')
		l1.grid(column=1, row=1)
		l2.grid(column=1, row=2)
		

		# All Entries
		in5 = Tkinter.Entry(frame, width=10)
		in6 = Tkinter.Entry(frame, width=10)
		in7 = Tkinter.Entry(frame, width=10)
		in8 = Tkinter.Entry(frame, width=10)


		in5.grid(column=2, row=1)
		in6.grid(column=2, row=2)
		in7.grid(column=3, row=1)
		in8.grid(column=3, row=2)

		# buttons widget
		Tkinter.Button(mainframe, text=' Refresh Size ', command=self.refreshsize , width=15).pack(side='top', padx=10, pady=10)
		Tkinter.Button(mainframe, text=' Lock Focus ', command=self.focusregion_lock , width=15).pack(side='top', padx=10, pady=10)
		Tkinter.Button(mainframe, text=' Bird Eye ', command=self.birdview , width=15).pack(side='top', padx=10, pady=10)


		# controls collections
		self.controls = [ in1, in2, in3, in4, in5, in6, in7, in8]

		return


		# focus regoin locking function
	def focusregion_lock(self):
		cx1 = self.controls[0].get() # x1
		cy1 = self.controls[1].get() # y1
		cx2 = self.controls[2].get() # x2
		cy2 = self.controls[3].get() # y2
		self.defaultimage = img = FocusRegion(self.defaultimage, x1=int(cx1), y1=int(cy1),x2=int(cx2), y2=int(cy2))
		self.load_default_values()
		return

		# bird view adjustment function
	def birdview(self, *args):
		print '[*] Bird View Adjustments.'
		# Get All X Points
		x1 = int(self.controls[4].get())
		
		# make it clear
		self.controls[5].delete(0, 'end')
		self.controls[5].insert(0, self.canvas_obj.imgwidth-x1)

		x2 = int(self.controls[5].get())
		
		y1 = int(self.controls[6].get())

		# make it clear
		self.controls[7].delete(0, 'end')
		self.controls[7].insert(0, y1)
		
		y2 = int(self.controls[7].get())




		if not (x1 and x2):
			print '[-] Please Provide X Coordinate Sets Correctly To Perform Bird View Transformation.'
			print '[-] Zeros or blanks Values not Allowed. Use `1` For Smallest Digit.'
			return

		if not (y1 and y2):
			print '[-] Please Provide Y Coordinate Sets Correctly To Perform Bird View Transformation.'
			print '[-] Zeros or blanks Values not Allowed. Use `1` For Smallest Digit.'
			return
			
		img = self.defaultimage # cvimage
		img, _ = EagleView(img, tl=[x1, y1], tr=[x2, y2])
		self.tkimage(img)
		self.setimage()
		return


		# image refresh function
	def refreshsize(self, *args):
		print '[*] Focus Region Adjustments.'
		tmp = self.defaultimage
		cx1 = self.controls[0].get() # x1
		cy1 = self.controls[1].get() # y1
		cx2 = self.controls[2].get() # x2
		cy2 = self.controls[3].get() # y2

		img = FocusRegion(tmp, x1=int(cx1), y1=int(cy1),x2=int(cx2), y2=int(cy2))
		self.tkimage(img)
		self.setimage()

		return

		# set image
	def setimage(self):
		self.canvas_obj.delete("all")
		self.canvas_obj.create_image(0,0, image=self.canvas_obj.imgref, anchor=Tkinter.NW)
		self.canvas_obj.config( height = self.canvas_obj.imgheight, width = self.canvas_obj.imgwidth )

		return


		# tkimage
	def tkimage(self, cvimg):
		self.canvas_obj.imgwidth = cvimg.shape[1]
		self.canvas_obj.imgheight = cvimg.shape[0]
		cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
		self.canvas_obj.imgref = ImageTk.PhotoImage(Image.fromarray(cvimg))
		return self.canvas_obj.imgref

		# create canvas
	def create_canvas(self):
		obj = Tkinter.Canvas(self, background='Red')
		obj.pack(side='top')
		return obj


# main function
def  main():
	# frame array
	frame = cv2.imread('../testing_data/test1.jpg')
	root = AdjustmentWindow(frame, className=' Adjustments Finding Window..')
	root.mainloop()
	return


# main trigger
if __name__ == '__main__':
	main()


