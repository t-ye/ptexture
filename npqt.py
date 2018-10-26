"""
Interoperability between NumPy arrays and PyQt images.
"""

def arr_to_image(arr, form) :


	from functools import partial
	import PyQt5.QtGui as QtGui
	import numpy as np

	imgen = partial(QtGui.QImage, arr, arr.shape[1], arr.shape[0])

	bitsPerPixel = 1

	if form == QtGui.QImage.Format_RGB888 :
		bitsPerPixel = 3

	return imgen(arr.shape[1] * bitsPerPixel, form)
