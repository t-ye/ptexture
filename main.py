from __future__ import annotations
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtGui as QtGui
import numpy as np
import npqt
import color
from color import channelsplit, hsv2rgb
from typing import Callable

#black = np.zeros((300, 300), dtype=np.uint8)
rows = 1080
cols = 1920

default_kwargs = {'R':rows, 'C':cols, 'zoom':4}
class Main(QtWidgets.QMainWindow) :

	sentinel = object()

	def __init__(self) :

		super().__init__()
		self.ptextures_dict = None
		self.curr_idx = None

		#self.curr_idx = None

		#self.bases = dict()
		self.ptextures = list()
		self.ptextures_args = dict()

		self.createWidgets()
		self.setupLayout()
		self.createPixmaps()

		self.show()

	def addpTexture(self, ptexture : ptexture.ptexture) :

		#base = ptexture.base
		#while base is not None :
		#	self.ptextures.add(base)
		#	base = base.base

		self.ptextures.append(ptexture)
		self.comboBox.addItem(str(self.comboBox.count()))

	def updateTexture(self, idx) :

		if self.curr_idx == idx :
			return
		if self.comboBox.currentIndex() != idx :
			self.comboBox.setCurrentIndex(idx)
			return
		self.curr_idx = idx

		texture = self.ptextures[idx]

		self.ptextures_dict = {key:None for key in texture.kwargs}
		self.ptextures_dict_types = {key:t for key,t in zip(texture.kwargs,
		texture.type_hints)}


		# clear vlayout
		for i in reversed(range(self.vlayout.count())):
			widget = self.vlayout.itemAt(i).widget()
			self.vlayout.removeWidget(widget)
			widget.setParent(None)
			widget.deleteLater()

		def buttonAction(kwarg) :
			def inner() :
				new, ok = \
				QtWidgets.QInputDialog.getText(self, '', 'Enter value of ' + kwarg,
				QtWidgets.QLineEdit.Normal, '')
				if ok :
					self.ptextures_dict[kwarg] = new
			return inner

		for kwarg in texture.kwargs :
			buttonAction(kwarg)()
			button = QtWidgets.QPushButton(kwarg)
			button.clicked.connect(buttonAction(kwarg))
			self.vlayout.addWidget(button)


		dct = {k:self.ptextures_dict_types[k](v) for k,v in self.ptextures_dict.items()}

		im = npqt.arr_to_image(*texture(**dct))



		pm = QtGui.QPixmap(im)
		self.label.setPixmap(pm)

	def createPixmaps(self) :

		self.pixmapStr = dict()

		from functools import partial
		#import ptexture
		from ptexture import noisefun
		import ptexture
		from partial_ext import partial_ext
		import color

		self.addpTexture(ptexture.noise)
		#self.addpTexture(ptexture.colornoise)

		self.updateTexture(0)
		#self.addTexture('noise', ptexture.noise)
		#self.addTexture('colornoise',
		#               partial_ext(noisefun, R=rows, C=cols,
		#							             fmt=color.rgb888))
		#self.addTexture('zoomed smooth noise', ptexture.zsn, base_name='noise')
		#self.addTexture('zoomed smooth colornoise',
		#                partial_ext(ptexture.zoomed_smooth_noise, zoom=4),
		#                base_name='colornoise')
		#self.addTexture('blur', blur, base_name='noise')
		#self.addTexture('colorblur', blur, base_name='colornoise')
		#self.addTexture('turbulence',
		#               partial_ext(turbulence, zoom=64),
		#							 base_name='noise')
		#self.addTexture('blueturbulence',
		#                lambda **kwargs : (kwargs['base'], kwargs['fmt']),
		#							  base_name='turbulence',
		#							  imgfy=npqt.g8_to_blue)
		#self.addTexture('colorturbulence',
		#                partial_ext(turbulence, zoom=64),
		#							  base_name='colornoise')
		#self.addTexture('marblebase',
		#                partial_ext(marble_base, R=rows, C=cols))
		#self.addTexture('marble',
		#                marble_true,
		#								base_name = 'noise')
		#self.addTexture('colormarble',
		#                marble_true,
		#								base_name = 'colornoise')
		#self.addTexture('wood base', partial_ext(wood_base, R=rows, C=cols))
		#self.addTexture('wood', wood, base_name='noise')
		#self.addTexture('brownwood',
		#                lambda **kwargs : (kwargs['base'], kwargs['fmt']),
		#                imgfy=npqt.g8_to_brown,
		#								base_name='wood')
		#self.addTexture('weird base', partial(weird_base, R=rows, C=cols),
		#imgfy=npqt.arr_to_reds)
		#self.addTexture('weird', partial(weird), base_name='noise',
		#imgfy=npqt.arr_to_reds)

		#self.updatePixmapStr('noise')

	def createWidgets(self) :
		self.vlayout = None
		self.label = QtWidgets.QLabel()
		self.label.setScaledContents(True) # auto resize

		#self.paramComboBox = QtWidgets.QComboBox(self)

		self.new_noise = QtWidgets.QPushButton('new noise')
		self.new_noise.clicked.connect(lambda : self.createPixmaps())
		self.next = QtWidgets.QPushButton('next')
		self.next.clicked.connect(lambda :
			self.comboBox.setCurrentIndex((self.comboBox.currentIndex() + 1) %
		                                self.comboBox.count()
			)
		)
		self.prev = QtWidgets.QPushButton('prev')
		self.prev.clicked.connect(lambda :
			self.comboBox.setCurrentIndex((self.comboBox.currentIndex() - 1) %
		                                self.comboBox.count()
			)
		)
		self.comboBox =  QtWidgets.QComboBox(self)
		#self.comboBox.currentTextChanged.connect(lambda s : self.updatePixmapStr(s))
		self.comboBox.currentIndexChanged.connect(lambda idx : self.updateTexture(idx))

	def setupLayout(self) :
		# setup mainLayout
		widget = QtWidgets.QWidget()
		widget3 = QtWidgets.QWidget()
		hlayout = QtWidgets.QHBoxLayout(widget)
		vlayout = QtWidgets.QVBoxLayout(widget3)
		self.mainLayout = hlayout
		self.setLayout(self.mainLayout)
		self.setCentralWidget(widget)

		# add to mainLayout

		vlayout.addWidget(self.new_noise)

		widget2 = QtWidgets.QWidget()
		changer = QtWidgets.QHBoxLayout(widget2)
		changer.addWidget(self.prev)
		changer.addWidget(self.comboBox)
		changer.addWidget(self.next)

		vlayout.addWidget(changer.parent())
		vlayout.addWidget(self.label)

		hlayout.addWidget(widget3)
		container = QtWidgets.QWidget()
		self.vlayout = QtWidgets.QVBoxLayout(container)
		self.mainLayout.addWidget(container)
		#container.hide()

	def updatePixmapStr(self, name, set_to=True) :

		if set_to :
			pm = self.pixmapStr[name]
			if callable(pm) :
				pm = pm()
				self.pixmapStr[name] = pm


			self.label.setPixmap(pm)
			self.pixmap_name = name
		else :

			pm = self.pixmapStr[name]
			if callable(pm) :
				pm = pm()
				self.pixmapStr[name] = pm

def blur(**kwargs) :

	base = kwargs['base']
	R,C=base.shape[:2]
	fmt = kwargs['fmt']
	# get matrix in middle same as base
	# but with a wraparound border around it
	repborder = np.tile(np.atleast_3d(base), (3,3,1))[R-1:2*R+1,C-1:2*C+1]
	#print(base.shape)
	#print(repborder.shape)

	# get the matrices that are a result of
	# moving the central instance of base left, right down, and up cast to 16 bits
	# to prevent later addition overflow
	l = repborder[1:R+1,:C].astype(np.uint16)
	r = repborder[1:R+1:,2:]
	b = repborder[2:,1:C+1]
	t = repborder[:R,1:C+1]

	#print(l.shape)
	#print(r.shape)
	#print(b.shape)
	#print(t.shape)

	return ((l+r+b+t+np.atleast_3d(base))/5, fmt)

def zoomed_smooth_noise(**kwargs) :

	base = kwargs['base']
	zoom = kwargs['zoom']
	fmt = kwargs['fmt']
	# m assumed 2D

	from time import time

	t = time()


	# get ranges corresonding to the top left (1/zoom)th
	# portion of the matrix
	(xf, yf), i = np.modf(np.indices(base.shape[:2]) / zoom)
	x, y = i.astype(np.int)
	print(time() - t)

	# up, left (negative indices allowed!)
	u = (x-1)
	l = (y-1)

	depth = 1 if base.ndim == 2 else base.shape[2]

	t = time()
	v  = (   xf  *    yf ).repeat(depth).reshape(base.shape) * base[x,y] \
	   + ((1-xf) *    yf ).repeat(depth).reshape(base.shape) * base[u,y] \
	   + (   xf  * (1-yf)).repeat(depth).reshape(base.shape) * base[x,l] \
	   + ((1-xf) * (1-yf)).repeat(depth).reshape(base.shape) * base[u,l]

	return (v, fmt)


def iterate(f, x) :
	yield x
	while 1:
		x = f(x)
		yield x


base_temp = None
fmt_temp = None

def zoomed_smooth_noise_helper(zoom) :
	global base_temp
	global fmt_temp
	return zoomed_smooth_noise(base=base_temp, zoom=zoom, fmt=fmt_temp)[0]


def turbulence(**kwargs) :

	from itertools import takewhile, accumulate
	from multiprocessing import Pool

	base = kwargs['base']
	zoom = kwargs['zoom']
	fmt = kwargs['fmt']

	izoom = zoom
	zooms = list(takewhile(lambda x : x >= 1, iterate(lambda x : x/2, zoom)))

	global base_temp
	global fmt_temp
	base_temp = base
	#print(base)
	fmt_temp = fmt
	#from time import time
	#t = time()
	#with Pool(len(zooms)) as p :
	#	vs = p.map(zoomed_smooth_noise_helper, zooms)
	#v = sum(v*zooms[i] for (i,v) in enumerate(vs))

	#print('Multiprocessing time: ', round(time() - t, 2))

	#t = time()
	v = np.zeros(base.shape, dtype=np.float)
	while zoom >= 1 :
		v += zoomed_smooth_noise(base=base, zoom=zoom, fmt=fmt)[0] * zoom
		zoom /= 2
	#print('Loop time: ', round(time() - t, 2))


	return (v / (2*izoom), fmt)


def arr_to_blues(arr) :
	narr = arr
	# go from grayscale to RGB
	# R = 0, G = 0
	arr  = np.full((*arr.shape, 3), 0, dtype=np.uint8)
	# set blue (invert)
	arr[...,2] = narr
	return npqt.arr_to_image(arr, color.rgb888)


def marble_base(**kwargs) :

	R,C = kwargs['R'], kwargs['C']

	arr = np.indices((R,C))
	xy = arr[0] + arr[1]
	return (128*(np.sin(xy / np.sqrt(max(R,C)))+1), color.gray8)

def marble_true(**kwargs) :

	base = kwargs['base']
	fmt = kwargs['fmt']
	R,C = base.shape[:2]
	x,y=np.indices((R,C))
	depth = 1 if len(base.shape) == 2 else base.shape[2]

	# period of sinusoidal
	xp = 5.0
	yp = 10.0

	# turbulence power
	power = 5.0
	zoom = 64.0

	xy = (x * xp / R + y * yp / C).repeat(depth).reshape(base.shape) \
	     + power * turbulence(zoom=zoom, base=base,fmt=fmt)[0] / 256
	v = 256 * np.abs(np.sin(xy * np.pi))

	return (v, fmt)

def wood_base(**kwargs) :

	R,C= kwargs['R'], kwargs['C']

	arr = np.indices((R,C), dtype=np.float)
	# move origin to center
	arr[0] -= R / 2
	arr[1] -= C / 2

	return (128 * (np.sin(np.sqrt(arr[0] ** 2 + arr[1] ** 2) / np.sqrt(R+C))+1),
          color.gray8)

def wood(**kwargs) :

	base = kwargs['base']
	fmt = kwargs['fmt']

	period = 12.0 # number of rings
	power = 0.05 # twists
	zoom = 32.0 # turbulence detail

	R,C = base.shape
	x,y = np.indices((R,C))

	x = (x - R / 2) / R
	y = (y - C / 2) / C
	d = np.sqrt(x**2 + y**2) + power * turbulence(zoom=zoom, base=base,
	fmt=fmt)[0] / 256
	return (128 * np.abs(np.sin(2 * period * d * np.pi)), fmt)

def weird_base(**kwargs) :

	R,C = kwargs['R'], kwargs['C']
	arr = np.indices((R,C))

	f = 64*(np.sin(arr[0] / np.pi**2)+1 + np.sin(arr[1] / np.pi**2)+1)

	return (f, color.gray8)

def weird(**kwargs) :

	base = kwargs['base']
	fmt = kwargs['fmt']

	zoom = 32
	power = 128

	R,C = base.shape
	x,y = np.indices((R,C), dtype=np.float)

	turb = turbulence(zoom=zoom, base=base, fmt=fmt)[0]
	x += power * turb / 256
	y += power * turb[::-1,::-1] / 256 # ???

	f = 64*(np.sin(x / np.pi**2)+1 + np.sin(y / np.pi**2)+1)

	return (f, fmt)


if __name__ == '__main__' :
	app = QtWidgets.QApplication([])

	window = Main()

	app.exec_()

#def addTexture(self, name : str,
#	                    generator : Callable[...,(np.ndarray,color.colorformat)],
#											base_name : str = None,
#											imgfy : Callable[[np.ndarray], QtGui.QImage] = None) :
#
#		# allow lazy generation of bases
#		self.bases[name] = (generator, None, None, base_name)
#		#                   generator array format base_name
#
#		def pixmapGenerator() :
#			nonlocal generator
#			nonlocal base_name
#			nonlocal imgfy
#
#			if imgfy is None : # default
#				imgfy = npqt.arr_to_image
#
#			if base_name is None : # no base
#				arr, fmt = generator(**default_kwargs)
#			else :
#				# get base info
#				base_generator, base, fmt, bb_name = self.bases[base_name]
#
#				if base is None : # bases needs to be generated
#
#					self.updatePixmapStr(base_name, set_to=False)
#
#				_, base, fmt, _ = self.bases[base_name]
#				arr, fmt = generator(base=base, fmt=fmt, **default_kwargs)
#
#			self.bases[name] = (generator, arr, fmt, base_name)
#
#			im = imgfy(arr, fmt) # ????
#			pixmap = QtGui.QPixmap(im)
#
#			return pixmap
#
#		self.pixmapStr[name] = pixmapGenerator
#		self.comboBox.addItem(name)
