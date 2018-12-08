from PyQt5 import QtWidgets
from .extension import clearLayout
from . import QNamedDict
import functional

class QNamedDictDemo(QtWidgets.QMainWindow) :

	def __init__(self) :

		super().__init__()

		# needs a dummy to paint on
		self.centralWidget = QtWidgets.QWidget()
		self.layout = QtWidgets.QVBoxLayout(self.centralWidget)
		self.setCentralWidget(self.centralWidget)

		self.show()

	def set(self, param : functional.StringParameter) :
		clearLayout(self.layout)
		self.dict_gui = QNamedDict(param, self)
		self.layout.addWidget(self.dict_gui)

		self.getButton = QtWidgets.QPushButton('Get')
		self.getButton.clicked.connect(lambda : print(self.get()))
		self.layout.addWidget(self.getButton)

		self.updateButton = QtWidgets.QPushButton('Update')
		self.updateButton.clicked.connect(lambda : print(self.update()))
		self.layout.addWidget(self.updateButton)


	def get(self) :

		return self.dict_gui.get()

	def update(self) :

		return self.dict_gui.update()

a = functional.StringParameter('a', ('1', '2', '3'))
b = functional.StringParameter('b', None)
d = functional.StringParameter('d', None)
e = functional.StringParameter('e', None)
f = functional.StringParameter('f', None)
c = functional.StringParameter('c', ('4',), (d,e,f))
A = functional.StringParameter('A', ('5', '6'), (a,b,c))

if __name__ == '__main__' :

	from sys import argv

	script, to_demo = argv
	app = QtWidgets.QApplication([])
	if to_demo == 'QNamedDict' :
		window = QNamedDictDemo()
		window.set(A)

	app.exec_()
