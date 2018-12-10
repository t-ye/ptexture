from PyQt5 import QtWidgets, QtGui, QtCore
import qt_ext
import ptexture
import npqt
import functional


QtCore.pyqtRemoveInputHook() # remove this when done debugging

class Main(QtWidgets.QMainWindow) :

	def __init__(self) :

		super().__init__()

		self.centralWidget = QtWidgets.QFrame()
		self.setCentralWidget(self.centralWidget)

		self.createDisplay()
		self.createParamGui()
		self.createLayout()
		self.populateLayout()

		self.attachSignals_paramGui()

		self.show()


	def createDisplay(self) :
		self.textureDisplay = QtWidgets.QLabel(self)
		self.textureDisplay.setScaledContents(True) # auto resize

	def setDisplay(self, pixmap : QtGui.QPixmap) :

		self.textureDisplay.setPixmap(pixmap)

	def updateDisplay(self) :

		tex = self.paramGui_dict.get()(**self.paramGui_dict.getDict()['texture'])
		im = ptexture.textureToImage(tex)
		pixmap = QtGui.QPixmap.fromImage(im)
		self.setDisplay(pixmap)

	def createParamGui(self) :

		self.param = functional.StringParameter('texture',
			choices    = tuple(ptexture.ptexture.instances.keys()),
			parse      = lambda choice : ptexture.ptexture.instances[choice],
			get_params = lambda ptex : ptex.params)

		self.paramGui_frame = QtWidgets.QFrame(self)
		self.paramGui_layout = QtWidgets.QVBoxLayout(self.paramGui_frame)

		self.paramGui_dict = qt_ext.QNamedDict(self.param, self.paramGui_frame)
		self.paramGui_updateButton = QtWidgets.QPushButton('Update')

		self.paramGui_layout.addWidget(self.paramGui_dict)
		self.paramGui_layout.addWidget(self.paramGui_updateButton)


	def attachSignals_paramGui(self) :

		self.paramGui_updateButton.clicked.connect(self.updateDisplay)


	def createLayout(self) :
		self.layout = QtWidgets.QHBoxLayout(self.centralWidget)

	def populateLayout(self) :
		self.layout.addWidget(self.textureDisplay)
		self.layout.addWidget(self.paramGui_frame)

if __name__ == '__main__' :
	app = QtWidgets.QApplication([])
	window = Main()
	app.exec_()
