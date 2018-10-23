import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtGui as QtGui
import numpy as np

#black = np.zeros((300, 300), dtype=np.uint8)
rows = 100
cols = 100

class Main(QtWidgets.QMainWindow) :

	def __init__(self) :

		super().__init__()


		self.createWidgets()
		self.createPixmaps()
		self.setupLayout()

		self.show()

	def addPixmap(self, generator, bw=True) :

		data = generator()
		if bw :
			data = data.astype(np.uint8)
			im = bw_image(data)
		else :
			pass

		pixmap = QtGui.QPixmap(im)
		self.pixmaps.append(pixmap)



	def createPixmaps(self) :
		self.pixmaps = []

		from functools import partial

		data = np.random.randint(256, size=(rows, cols), dtype=np.uint8)
		self.noise = data

		self.addPixmap(lambda : data)
		self.addPixmap(partial(zoomed_smooth_noise, data, 4))
		self.addPixmap(partial(blur, data))
		self.addPixmap(partial(turbulence, data, 64))
		self.addPixmap(partial(marble_base, rows))
		self.addPixmap(partial(marble_true, data))

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
		self.label.setPixmap(self.pixmaps[self.pixmap_idx])

	def cyclePixmap(self) :
		self.updatePixmap((self.pixmap_idx+1) % len(self.pixmaps))

def blur(mat) :
	R,C=mat.shape
	# get matrix in middle same as mat
	# but with a wraparound border around it
	repborder = np.tile(mat, (3,3))[R-1:2*R+1,C-1:2*C+1]

	# get the matrices that are a result of
	# moving the central instance of mat left, right
	# down, and up
	# cast to 16 bits to prevent later addition overflow
	l = repborder[1:R+1,:C].astype(np.uint16)
	r = repborder[1:R+1:,2:]
	b = repborder[2:,1:C+1]
	t = repborder[:R,1:C+1]


	return ((l+r+b+t)//4).astype(np.uint8)

def zoomed_smooth_noise(m, zoom) :

	# m assumed 2D
	R,C = m.shape

	idxs = np.arange(R, dtype=np.float).repeat(C).reshape(R,C)

	# get ranges corresonding to the top left (1/zoom)th
	# portion of the matrix
	f, i = np.modf(idxs / zoom)
	i = i.astype(np.int)
	x, y = i, i.	T
	xf, yf = f, f.T


	# up, left (negative indices allowed!)
	u = (x-1)
	l = (y-1)

	v  =    xf  *    yf  * m[x,y]
	v += (1-xf) *    yf  * m[u,y]
	v +=    xf  * (1-yf) * m[x,l]
	v += (1-xf) * (1-yf) * m[u,l]

	return v

def turbulence(m, size) :

	v = np.zeros_like(m,dtype=np.float)
	isize = size

	while size >= 1 :
		v += zoomed_smooth_noise(m, size) * size
		size /= 2

	return v / (2*isize)


def bw_image(arr) :
	return QtGui.QImage(arr[...,np.newaxis], arr.shape[1], arr.shape[0], arr.shape[1],
	QtGui.QImage.Format_Grayscale8)

def blueify(arr) :
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
	arr = np.arange(R).repeat(C).reshape((R,C))
	x = arr
	y = arr.T

	# period of sinusoidal
	xp = 5.0
	yp = 10.0

	# turbulence power
	power = 5.0
	size = 64.0

	xy = x * xp / R + y * yp / C + power * turbulence(noise, size) / 256
	#v = 128 * (1+np.sin(xy * np.pi))
	v = 256 * np.abs(np.sin(xy * np.pi))

	return v

def bw_image(arr) :
	return QtGui.QImage(arr, arr.shape[1], arr.shape[0], arr.shape[1],
	QtGui.QImage.Format_Grayscale8)

def blueify(arr) :
	narr = arr
	# go from grayscale to RGB
	# R = 0, G = 0
	arr  = np.full((*narr.shape, 3), 0, dtype=np.uint8)
	# set B
	arr[...,2] = 255-narr
	return QtGui.QImage(arr, arr.shape[1], arr.shape[0], arr.shape[1]*3,
	QtGui.QImage.Format_RGB888)

if __name__ == '__main__' :
	app = QtWidgets.QApplication([])

	#label = QtWidgets.QLabel('Hello World!')
	window = Main()

	#data = QtGui.QPixmap()
	#data.loadFromData(black, 300*300)
	#label.setPixmap(data)



	#label.show()


	app.exec_()

