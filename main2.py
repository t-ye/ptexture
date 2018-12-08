from PyQt5 import QtWidgets, QtGui
import qt_ext
import ptexture
import QNamedDict


class Main(QtWidgets.QMainWindow) :

	def __init__(self) :

		super().__init__()

		self.createWidgets()
		self.createLayout()


	def createWidgets(self) :
		self.display = QtWidgets.QLabel(self)
		self.label.setScaledContents(True) # auto resize

		self.param = functional.Parameter('texture', ptexture.ptexture.instances)

		self.param_gui = QNamedDict.QNamedDict(self.param)
