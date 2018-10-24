import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtGui as QtGui
import numpy as np

#black = np.zeros((300, 300), dtype=np.uint8)
rows = 1080
cols = 1920

class Main(QtWidgets.QMainWindow) :

	sentinel = object()

	def __init__(self) :

		super().__init__()


		self.createWidgets()
		self.createPixmaps()
		self.setupLayout()

		self.show()

	def addPixmap(self, generator, base=sentinel, imgfy=None) :

		def pixmapGenerator() :
			nonlocal base
			nonlocal generator
			nonlocal imgfy

			if base is Main.sentinel :
				base = lambda : self.base

			if imgfy is None :
				imgfy = arr_to_bwim

			arr = generator(base()) if base is not None else generator()

			im = imgfy(arr.astype(np.uint8))
			pixmap = QtGui.QPixmap(im)

			return pixmap

		self.pixmaps.append(pixmapGenerator)

	def createPixmaps(self) :

		self.pixmaps = []

		from functools import partial

		data = np.random.randint(256, size=(rows, cols), dtype=np.uint8)
		self.noise = data

		self.base = data

		self.addPixmap(lambda x : x)
		self.addPixmap(partial(zoomed_smooth_noise, 4))
		self.addPixmap(blur)
		self.addPixmap(partial(turbulence, 64))
		self.addPixmap(partial(turbulence, 64), imgfy=arr_to_blues)
		self.addPixmap(partial(marble_base, rows), None)
		self.addPixmap(marble_true)

		self.updatePixmap(0)

	def createWidgets(self) :
		self.label = QtWidgets.QLabel()
		self.label.setScaledContents(True) # auto resize

		self.button1 = QtWidgets.QPushButton('new noise')
		self.button1.clicked.connect(lambda : self.createPixmaps())
		self.button2 = QtWidgets.QPushButton('cycle')
		self.button2.clicked.connect(lambda : self.cyclePixmap())

	def setupLayout(self) :
		# setup layout
		widget = QtWidgets.QWidget()
		self.layout = QtWidgets.QVBoxLayout(widget)
		self.setLayout(self.layout)
		self.setCentralWidget(widget)

		# add to layout
		self.layout.addWidget(self.button1)
		self.layout.addWidget(self.button2)
		self.layout.addWidget(self.label)

	def updatePixmap(self, idx) :
		self.pixmap_idx = idx

		# eval
		pm = self.pixmaps[self.pixmap_idx]()
		self.pixmaps[self.pixmap_idx] = lambda : pm

		self.label.setPixmap(pm)

	def cyclePixmap(self) :
		self.updatePixmap((self.pixmap_idx+1) % len(self.pixmaps))

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

def marble_base(n) :

	arr = np.arange(n).repeat(n).reshape((n,n))
	arr = arr + arr.T

	return (128*(np.sin(arr)+1)).astype(np.uint8)

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

def arr_to_bwim(arr) :
	return QtGui.QImage(arr, arr.shape[1], arr.shape[0], arr.shape[1],
	QtGui.QImage.Format_Grayscale8)

def arr_to_blues(arr) :
	narr = arr
	# go from grayscale to RGB
	# R = 0, G = 0
	arr = np.full((*narr.shape, 3), 0, dtype=np.uint8)
	# set B
	arr[...,2] = narr
	return QtGui.QImage(arr, arr.shape[1], arr.shape[0], arr.shape[1]*3,
	QtGui.QImage.Format_RGB888)

if __name__ == '__main__' :
	app = QtWidgets.QApplication([])

	window = Main()

	app.exec_()

