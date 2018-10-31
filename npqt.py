"""
Interoperability between NumPy arrays and PyQt images.
"""


def arr_to_image(arr, fmt) :

	from functools import partial
	import PyQt5.QtGui as QtGui
	import numpy as np

	if arr.dtype != np.uint8 :
		arr = arr.astype(np.uint8)

	imgen = partial(QtGui.QImage, arr, arr.shape[1], arr.shape[0])


	bitsPerPixel = 1

	if fmt == QtGui.QImage.Format_RGB888 :
		bitsPerPixel = 3

	return imgen(arr.shape[1] * bitsPerPixel, fmt)

def g8_to_blue(arr,fmt) :

	from color import channelsplit

	import numpy as np
	import PyQt5.QtGui as QtGui

	depth = 1 if len(arr.shape) == 2 else arr.shape[2]
	if depth != 1 :
		raise ValueError('Depth of array must be 1')
	arr = channelsplit(lambda x : (
			np.zeros(x.shape),
			np.zeros(x.shape),
			arr
		), arr)
	return arr_to_image(arr, QtGui.QImage.Format_RGB888)

def g8_to_brown(arr,fmt) :

	from color import channelsplit

	import numpy as np
	import PyQt5.QtGui as QtGui

	depth = 1 if len(arr.shape) == 2 else arr.shape[2]
	if depth != 1 :
		raise ValueError('Depth of array must be 1')
	arr = channelsplit(lambda x : (
			np.clip(x+80, 0, 255),
			np.clip(x+30, 0, 255),
			np.full(x.shape, 30)
		), arr)
	return arr_to_image(arr, QtGui.QImage.Format_RGB888)

def arr_to_reds(arr,fmt) :

	import numpy as np
	from color import channelsplit, hsv2rgb
	import PyQt5.QtGui as QtGui


	# 8-bit space to degrees, then all colors to shades of red/yellow
	arr = channelsplit(lambda x : (
			360/256 * x * 60/360,
			np.full(x.shape, 1.0),
			np.full(x.shape, 1.0)
		), arr)
	arr = hsv2rgb(arr)
	return arr_to_image(arr, QtGui.QImage.Format_RGB888)
