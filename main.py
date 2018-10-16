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

	def createPixmaps(self) :
		self.pixmaps = []

		self.noise = np.random.randint(256, size=(rows, cols), dtype=np.uint8)
		qim = bw_image(self.noise)
		qpm = QtGui.QPixmap(qim)
		self.pixmaps.append(qpm)

		smoothed = blur(self.noise)
		qim = bw_image(smoothed)
		qpm = QtGui.QPixmap(qim)
		self.pixmaps.append(qpm)


		from functools import partial
		zoomedSmooth = np.fromiter(map(partial(smooth_noise, self.noise),
			map(lambda xy : (xy[0]/8, xy[1]/8), np.ndindex(self.noise.shape))), dtype=np.uint8)
		zoomedSmooth = zoomedSmooth.reshape(self.noise.shape)
		qim = bw_image(zoomedSmooth)
		qpm = QtGui.QPixmap(qim)
		self.pixmaps.append(qpm)


		turbulent = np.fromiter(map(partial(turbulence, self.noise, 64),
			np.ndindex(self.noise.shape)), dtype=np.uint8)
		turbulent = turbulent.reshape(self.noise.shape)
		qim = bw_image(turbulent)
		qpm = QtGui.QPixmap(qim)
		self.pixmaps.append(qpm)

		turbulent_blue = blueify(turbulent)
		qpm = QtGui.QPixmap(turbulent_blue)
		self.pixmaps.append(qpm)

		"""
		from functools import partial
		zoomed = np.fromiter(map(partial(blur_zoom, self.noise, 8),
		np.ndindex(self.noise.shape)), dtype=np.uint8)
		zoomed = np.reshape(zoomed, self.noise.shape)
		qim = bw_image(zoomed)
		qpm = QtGui.QPixmap(qim)
		self.pixmaps.append(qpm)
		"""

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

def smooth_noise(m,xy) :

	R,C = m.shape
	x,y=xy

	fx = x - int(x)
	fy = y - int(y)


	x1 = (int(x) + R) % R
	y1 = (int(y) + C) % C

	x2 = (x1 + R - 1) % R
	y2 = (y1 + C - 1) % C

	v = 0
	v += fx * fy * m[x1,y1]
	v += (1-fx) * fy * m[x2,y1]
	v += fx * (1-fy) * m[x1,y2]
	v += (1-fx) * (1-fy) * m[x2,y2]

	return int(v)

def turbulence(m, size, xy) :

	v = 0
	isize = size

	x,y=xy

	while size >= 1 :
		v += smooth_noise(m, (x/size, y/size)) * size
		size /= 2

	return int(v / (2*isize))



#smoothed = np.empty((rows, cols))
#for idxs in np.ndindex(rows, cols) :
#	smoothed[idxs] = smooth(*idxs)


#smoothed = np.fromiter(map(smooth, np.ndindex(rows, cols)), dtype=np.uint8)
#smoothed = np.reshape(smoothed, (rows, cols))


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

