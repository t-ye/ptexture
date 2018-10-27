import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtGui as QtGui
import numpy as np
import npqt
from color import channelsplit, hsv2rgb
from typing import Callable

#black = np.zeros((300, 300), dtype=np.uint8)
rows = 1080
cols = 1920

class Main(QtWidgets.QMainWindow) :

	sentinel = object()

	def __init__(self) :

		super().__init__()


		self.bases = dict()

		self.createWidgets()
		self.createPixmaps()
		self.setupLayout()

		self.show()

	def addTexture(self, name : str,
	                    generator : Callable[...,np.ndarray],
											base_name : str = None,
											imgfy : Callable[[np.ndarray], QtGui.QImage] = None) :

		# allow lazy generation of bases
		self.bases[name] = (generator, None, None, base_name)
		#                   generator array format base_name

		def pixmapGenerator() :
			nonlocal generator
			nonlocal base_name
			nonlocal imgfy

			if imgfy is None : # default
				imgfy = npqt.arr_to_image

			if base_name is None : # no base
				arr, fmt = generator()
			else :
				# get base info
				base_generator, base, fmt, base_base_name = self.bases[base_name]

				if base is None : # bases needs to be generated
					#stack = []
					#stack.append(base_name)
					#while base_base_name is not None and \
					#      self.bases[base_base_name][1] is None :
					#	stack.append(base_base_name)
					#	base_generator, base, fmt, base_base_name = self.bases[base_base_name]
					#while stack :
					#	el = stack.pop()
					#	base_generator, _, _, base_base_name = self.bases[el]
					#	self.bases[el] = (base_generator,
					#	*base_generator(base=),
					#		base_base_name)

					if base_base_name is not None :
						bb_generator, bb, fmt, bbb_name = self.bases[base_base_name]
						if bb is None :
							self.bases[bb_name] = (bb_generator, *bb_generator(fmt=fmt), fmt, bbb_name)

					base_generator, base, fmt, base_base_name = self.bases[base_name]
					self.bases[base_name] = (base_generator, *base_generator(base=bb,
					fmt=fmt),
					base_base_name)
				base_generator, base, fmt, base_base_name = self.bases[base_name]
				arr, fmt = generator(base=base, fmt=fmt)

			self.bases[name] = (generator, arr, fmt, base_name)

			im = imgfy(arr, fmt) # ????
			pixmap = QtGui.QPixmap(im)

			return pixmap

		#self.pixmaps.append(pixmapGenerator)
		self.pixmapStr[name] = pixmapGenerator
		self.comboBox.addItem(name)

	def createPixmaps(self) :

		#self.pixmaps = []
		self.pixmapStr = dict()

		from functools import partial
		import ptexture
		from texturers import noisefun
		from partial_ext import partial_ext

		self.addTexture('noise', partial_ext(noisefun, R=rows, C=cols))
		self.addTexture('colornoise',
		               partial_ext(noisefun, R=rows, C=cols,
									             fmt=QtGui.QImage.Format_RGB888))
		self.addTexture('zoomed smooth noise',
		                partial_ext(zoomed_smooth_noise, zoom=4),
		                base_name='noise')
		self.addTexture('zoomed smooth colornoise',
		                partial_ext(zoomed_smooth_noise, zoom=4),
		                base_name='colornoise')
		self.addTexture('blur', blur, base_name='noise')
		self.addTexture('colorblur', blur, base_name='colornoise')
		self.addTexture('turbulence',
		               partial_ext(turbulence, zoom=64),
									 base_name='noise')
		self.addTexture('blueturbulence',
		                lambda **kwargs : (kwargs['base'], kwargs['fmt']),
									  base_name='turbulence',
									  imgfy=npqt.g8_to_blue)
		self.addTexture('colorturbulence',
		                partial_ext(turbulence, zoom=64),
									  base_name='colornoise')
		self.addTexture('marblebase',
		                partial_ext(marble_base, R=rows, C=cols))
		self.addTexture('marble',
		                marble_true,
										base_name = 'noise')
		self.addTexture('colormarble',
		                marble_true,
										base_name = 'colornoise')
		self.addTexture('wood base', partial_ext(wood_base, R=rows, C=cols))
		self.addTexture('wood', wood, base_name='noise')
		self.addTexture('brownwood',
		                lambda **kwargs : (kwargs['base'], kwargs['fmt']),
		                imgfy=npqt.g8_to_brown,
										base_name='wood')
		self.addTexture('weird base', partial(weird_base, R=rows, C=cols),
		imgfy=npqt.arr_to_reds)
		self.addTexture('weird', partial(weird), base_name='noise',
		imgfy=npqt.arr_to_reds)

		self.updatePixmapStr('noise')

	def createWidgets(self) :
		self.label = QtWidgets.QLabel()
		self.label.setScaledContents(True) # auto resize

		self.new_noise = QtWidgets.QPushButton('new noise')
		self.new_noise.clicked.connect(lambda : self.createPixmaps())
		self.next = QtWidgets.QPushButton('next')
		self.next.clicked.connect(lambda : self.updatePixmap(self.pixmap_idx + 1))
		self.prev = QtWidgets.QPushButton('prev')
		self.prev.clicked.connect(lambda : self.updatePixmap(self.pixmap_idx - 1))
		self.comboBox =  QtWidgets.QComboBox(self)
		self.comboBox.currentTextChanged.connect(lambda s : self.updatePixmapStr(s))

	def setupLayout(self) :
		# setup layout
		widget = QtWidgets.QWidget()
		self.layout = QtWidgets.QVBoxLayout(widget)
		self.setLayout(self.layout)
		self.setCentralWidget(widget)

		# add to layout

		self.layout.addWidget(self.new_noise)

		widget2 = QtWidgets.QWidget()
		changer = QtWidgets.QHBoxLayout(widget2)
		changer.addWidget(self.prev)
		changer.addWidget(self.comboBox)
		changer.addWidget(self.next)

		self.layout.addWidget(changer.parent())
		self.layout.addWidget(self.label)

	def updatePixmapStr(self, name) :
		#pm = self.pixmapStr[name]

		if self.comboBox.currentText() != name :
			self.comboBox.setCurrentText(name)
			return

		self.pixmap_name = name

		pm = self.pixmapStr[name]()
		self.pixmapStr[self.pixmap_name] = lambda : pm

		self.label.setPixmap(pm)

	#def updatePixmap(self, idx) :
	#	idx %= len(self.pixmaps)

	#	if self.comboBox.currentIndex() != idx :
	#		self.comboBox.setCurrentIndex(idx)
	#		return

	#	self.pixmap_idx = idx

	#	# eval
	#	pm = self.pixmaps[self.pixmap_idx]()
	#	self.pixmaps[self.pixmap_idx] = lambda : pm

	#	self.label.setPixmap(pm)

def colornoise(rows, cols) :
	return np.random.randint(256, size=(rows, cols, 3), dtype=np.uint8)


def blur(**kwargs) :

	base = kwargs['base']
	R,C=base.shape[:2]
	fmt = kwargs['fmt']
	# get matrix in middle same as base
	# but with a wraparound border around it
	repborder = np.tile(np.atleast_3d(base), (3,3,1))[R-1:2*R+1,C-1:2*C+1]
	print(base.shape)
	print(repborder.shape)

	# get the matrices that are a result of
	# moving the central instance of base left, right down, and up cast to 16 bits
	# to prevent later addition overflow
	l = repborder[1:R+1,:C].astype(np.uint16)
	r = repborder[1:R+1:,2:]
	b = repborder[2:,1:C+1]
	t = repborder[:R,1:C+1]

	print(l.shape)
	print(r.shape)
	print(b.shape)
	print(t.shape)

	return ((l+r+b+t+np.atleast_3d(base))/5, fmt)

def zoomed_smooth_noise(**kwargs) :

	base = kwargs['base']
	zoom = kwargs['zoom']
	fmt = kwargs['fmt']
	# m assumed 2D

	idxs = np.indices(base.shape[:2])

	# get ranges corresonding to the top left (1/zoom)th
	# portion of the matrix
	f, i = np.modf(idxs / zoom)
	x, y = i.astype(np.int)
	xf, yf = f


	# up, left (negative indices allowed!)
	u = (x-1)
	l = (y-1)

	depth = 1 if len(base.shape) == 2 else base.shape[2]

	v  = (   xf  *    yf ).repeat(depth).reshape(base.shape) * base[x,y]
	v += ((1-xf) *    yf ).repeat(depth).reshape(base.shape) * base[u,y]
	v += (   xf  * (1-yf)).repeat(depth).reshape(base.shape) * base[x,l]
	v += ((1-xf) * (1-yf)).repeat(depth).reshape(base.shape) * base[u,l]

	return (v, fmt)

def turbulence(**kwargs) :

	base = kwargs['base']
	zoom = kwargs['zoom']
	fmt = kwargs['fmt']

	v = np.zeros_like(base,dtype=np.float)
	izoom = zoom

	while zoom >= 1 :
		v += zoomed_smooth_noise(base=base, zoom=zoom, fmt=fmt)[0] * zoom
		zoom /= 2

	return (v / (2*izoom), fmt)


def arr_to_bwim(arr) :
	return QtGui.QImage(arr[...,np.newaxis], arr.shape[1], arr.shape[0], arr.shape[1],
	QtGui.QImage.Format_Grayscale8)

def arr_to_blues(arr) :
	narr = arr
	# go from grayscale to RGB
	# R = 0, G = 0
	arr  = np.full((*arr.shape, 3), 0, dtype=np.uint8)
	# set blue (invert)
	arr[...,2] = narr
	return npqt.arr_to_image(arr, QtGui.QImage.Format_RGB888)


def marble_base(**kwargs) :

	R,C = kwargs['R'], kwargs['C']

	arr = np.indices((R,C))
	xy = arr[0] + arr[1]
	return (128*(np.sin(xy / np.sqrt(max(R,C)))+1), QtGui.QImage.Format_Grayscale8)

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
          QtGui.QImage.Format_Grayscale8)

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

	return (f, QtGui.QImage.Format_Grayscale8)

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

