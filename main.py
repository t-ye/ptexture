import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtGui as QtGui
import numpy as np
from hsv import hsv2rgb
from npqt import arr_to_image
from color import channelsplit

#black = np.zeros((300, 300), dtype=np.uint8)
rows = 1090
cols = 1920

class Main(QtWidgets.QMainWindow) :

	sentinel = object()

	def __init__(self) :

		super().__init__()


		self.createWidgets()
		self.createPixmaps()
		self.setupLayout()

		self.show()

	#def addTexture(self, name, texture, imgfy=None) :

	def addPixmap(self, name, generator, base=sentinel, imgfy=None) :

		def pixmapGenerator() :
			nonlocal base
			nonlocal generator
			nonlocal imgfy

			if base is Main.sentinel :
				base = lambda : self.base

			if imgfy is None :
				imgfy = arr_to_bandw

			arr = generator(base()) if base is not None else generator()

			im = imgfy(arr.astype(np.uint8))
			pixmap = QtGui.QPixmap(im)

			return pixmap

		self.pixmaps.append(pixmapGenerator)
		self.comboBox.addItem(name)

	def createPixmaps(self) :

		self.pixmaps = []

		from functools import partial
		import ptexture
		import texturers

		data = np.random.randint(256, size=(rows, cols), dtype=np.uint8)
		self.noise = data

		self.base = data

		self.addPixmap('noise', lambda x : x)
		# noise = ptexture.ptexture(texturers.noise_uint8)
		# wood = ptexture.ptexture(texturers.wood)
		# self.addTexture(noise)
		# self.addTexture(wood)
		self.addPixmap('zoomed smoothed noise', partial(zoomed_smooth_noise, 4))
		self.addPixmap('blur', blur)
		self.addPixmap('turbulence', partial(turbulence, 64))
		self.addPixmap('blue turbulence', partial(turbulence, 64), imgfy=arr_to_blues)
		self.addPixmap('marble base', partial(marble_base, rows, cols), None)
		self.addPixmap('marble', marble_true)
		self.addPixmap('wood base', partial(wood_base, rows, cols), None)
		self.addPixmap('wood', wood, imgfy=arr_to_browns)
		self.addPixmap('wood 2', wood, imgfy=arr_to_browns_naive)
		self.addPixmap('weird base', partial(weird_base, rows, cols), None, imgfy=arr_to_reds)

		self.updatePixmap(0)

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
		self.comboBox.currentIndexChanged.connect(lambda i : self.updatePixmap(i))

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

	def updatePixmap(self, idx) :

		if self.comboBox.currentIndex() != idx :
			self.comboBox.setCurrentIndex(idx)
			return

		self.pixmap_idx = idx

		# eval
		pm = self.pixmaps[self.pixmap_idx]()
		self.pixmaps[self.pixmap_idx] = lambda : pm

		self.label.setPixmap(pm)


def blur(mat) :
	R,C=mat.shape
	# get matrix in middle same as mat
	# but with a wraparound border around it
	repborder = np.tile(mat, (3,3))[R-1:2*R+1,C-1:2*C+1]

	# get the matrices that are a result of
	# moving the central instance of mat left, right down, and up cast to 16 bits
	# to prevent later addition overflow
	l = repborder[1:R+1,:C].astype(np.uint16)
	r = repborder[1:R+1:,2:]
	b = repborder[2:,1:C+1]
	t = repborder[:R,1:C+1]


	return ((l+r+b+t)//4).astype(np.uint8)

def zoomed_smooth_noise(zoom, m) :

	# m assumed 2D
	R,C = m.shape

	#idxs = np.arange(R, dtype=np.float).repeat(C).reshape(R,C)
	idxs = np.indices(m.shape)

	# get ranges corresonding to the top left (1/zoom)th
	# portion of the matrix
	f, i = np.modf(idxs / zoom)
	x, y = i.astype(np.int)
	xf, yf = f


	# up, left (negative indices allowed!)
	u = (x-1)
	l = (y-1)

	v  =    xf  *    yf  * m[x,y]
	v += (1-xf) *    yf  * m[u,y]
	v +=    xf  * (1-yf) * m[x,l]
	v += (1-xf) * (1-yf) * m[u,l]

	return v

def turbulence(size, m) :

	v = np.zeros_like(m,dtype=np.float)
	isize = size

	while size >= 1 :
		v += zoomed_smooth_noise(size, m) * size
		size /= 2

	return v / (2*isize)


def arr_to_bwim(arr) :
	return QtGui.QImage(arr[...,np.newaxis], arr.shape[1], arr.shape[0], arr.shape[1],
	QtGui.QImage.Format_Grayscale8)

def arr_to_blues(arr) :
	narr = arr
	# go from grayscale to RGB
	# R = 0, G = 0
	arr  = np.full((*narr.shape, 3), 0, dtype=np.uint8)
	# set blue (invert)
	arr[...,2] = arr
	return QtGui.QImage(arr, arr.shape[1], arr.shape[0], arr.shape[1]*3,
	QtGui.QImage.Format_RGB888)

def marble_base(R,C) :

	arr = np.indices((R,C))
	xy = arr[0] + arr[1]
	return (128*(np.sin(xy)+1)).astype(np.uint8)

def marble_true(noise) :

	R,C = noise.shape
	x,y=np.indices((R,C))

	# period of sinusoidal
	xp = 5.0
	yp = 10.0

	# turbulence power
	power = 5.0
	size = 64.0

	xy = x * xp / R + y * yp / C + power * turbulence(size, noise) / 256
	#v = 128 * (1+np.sin(xy * np.pi))
	v = 256 * np.abs(np.sin(xy * np.pi))

	return v

def wood_base(R,C) :

	arr = np.indices((R,C), dtype=np.float)
	# move origin to center
	arr[0] -= R / 2
	arr[1] -= C / 2

	return 128 * (np.sin(np.sqrt(arr[0] ** 2 + arr[1] ** 2) / np.sqrt(R+C))+1)

def wood(noise) :

	period = 12.0 # number of rings
	power = 0.05 # twists
	size = 32.0 # turbulence detail

	R,C = noise.shape
	x,y = np.indices((R,C))

	x = (x - R / 2) / R
	y = (y - C / 2) / C
	d = np.sqrt(x**2 + y**2) + power * turbulence(size, noise) / 256
	return 128 * np.abs(np.sin(2 * period * d * np.pi))

def weird_base(R, C) :

	arr = np.indices((R,C))

	f = 64*(np.sin(arr[0] / np.pi)+1 + np.sin(arr[1] / np.pi)+1)

	return f

def arr_to_bandw(arr) :
	return QtGui.QImage(arr, arr.shape[1], arr.shape[0], arr.shape[1],
	QtGui.QImage.Format_Grayscale8)

def arr_to_reds(arr) :
	narr = arr
	arr = np.zeros((*narr.shape, 3), dtype=np.float)
	# 8-bit space to degrees, then all colors to shades of red/yellow
	arr[...,0] = 360/256 * narr * 60/360
	#arr[...,0] = 60/256 * narr
	arr[...,1] = 1.0
	arr[...,2] = 1.0
	arr = hsv2rgb(arr)
	return arr_to_image(arr, QtGui.QImage.Format_RGB888)


def arr_to_blues(arr) :
	# go from grayscale to RGB
	# R = 0, G = 0
	# set B
	arr = channelsplit(lambda x : (np.full(x.shape, 0), np.full(x.shape, 0), arr),
	                   arr)
	return arr_to_image(arr, QtGui.QImage.Format_RGB888)

def arr_to_browns(arr) :
	narr = np.copy(arr)
	print(np.any(narr < 0))
	arr = channelsplit(lambda x : (
		 	np.clip(np.add(x, 80, dtype=np.int16), 0, 255),
			np.clip(np.add(x, 30, dtype=np.int16), 0, 255),
			np.full(x.shape, 30)
		), narr)
	arr2 = channelsplit(lambda x : (
			x + 80,
			x + 30,
			np.full(x.shape, 30)
		), narr)
	print(np.all(arr == arr2))

	return arr_to_image(arr, QtGui.QImage.Format_RGB888)

def arr_to_browns_naive(arr) :
	arr = channelsplit(lambda x : (
			x + 80,
			x + 30,
			np.full(x.shape, 30)
		), arr)

	return arr_to_image(arr, QtGui.QImage.Format_RGB888)


if __name__ == '__main__' :
	app = QtWidgets.QApplication([])

	window = Main()

	app.exec_()

