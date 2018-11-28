from PyQt5 import QtWidgets
import typing

@dataclasses.dataclass(frozen=True)
class gui_param :
	T : type
	#widget : typing.Callable[T, QtWidgets.QWidget]

@dataclasses.dataclass(frozen=True)
class ptexture_param() :
	name : str
	T : type
	widget : Callable[Qtidgets.QWidget]

class QParameterSelector(QtWidgets.QWidget) :

	def __init__(self, name) :

		self.layout = QtWidgets.QHBoxLayout(self)

		self.name = QtWidgets.QLabel(name, self)
		self.curr = QtWidgets.QLabel('', self)
		self.curr_data = None
		self.new = QtWidgets.QLineEdit('', self)

		self.layout.addWidget(self.name)
		self.layout.addWidget(self.curr)
		self.layout.addWidget(self.new)

	def get(self) :

		new_data = self.new.text()
		if new_data == '' : # no-op
			return {self.curr_data

		self.curr.setText(new_data)
		self.new.setText('')

		return self.curr_data

class QParameterDisplay(QtWidgets.QWidget) :

		self.layout.addWidget(widget)

	def __init__(self, name, children_names = None) :

		self.layout = QtWidgets.QVBoxLayout(self)

		self.selector = QParameterSelector(name)
		self.layout.addWidget(self.selector)

		if children_names is None :
			children_names = []

		self.children = []
		for child_name in children_names :
			child = QParameterDisplay(child_name)
			self.children.append(child)
			self.layout.addWidget(child)


class QParameterDisplay_ptexture(QtWidgets.QWidget) :

	def __init__(self, name) :
		self.name = QtWidgets.QLabel(name, self)
		self.curr = QtWidgets.QLabel('', self)
		self.curr_data = None
		self.new = QtWidgets.QLineEdit('', self)

	def get(self) :

		new_data = self.new.text()
		if new_data == '' : # no-op
			return {self.curr_data

		self.curr.setText(new_data)
		self.new.setText('')

		return self.curr_data
int_param = gui_param(int, ParamWidget)
