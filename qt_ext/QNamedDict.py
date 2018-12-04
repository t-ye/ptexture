import dataclasses
import typing
from .extension import widgetsOf
from PyQt5 import QtWidgets
import functional

class NamedDict(dict) :
	"""
	Dictionary with a name attribute.

	Can be used when the name is a function and the entries are keyword arguments.
	"""

	def __init__(self, name, *args, **kwargs) :

		super().__init__(*args, **kwargs)
		self.name = name

	def __repr__(self) :
		return f'NamedDict(name={self.name}, {super().__repr__()})'

def of(k : functional.StringParameter, parent) :
	return QNamedDict(k, parent)

class QNamedDict(QtWidgets.QWidget) :

	class Subentries(QtWidgets.QWidget) :

		def __init__(self, param, parent) :

			super().__init__(parent)

			self.parent = parent
			del parent
			self.param = param
			del param

			self.layout = QtWidgets.QVBoxLayout(self)
			for param in self.param.children :
				self.layout.addWidget(QNamedDict(param, self))

		def get(self) :
			dct = dict()
			for child in widgetsOf(self.layout) :
				dct.update(child.get())
			return dct

		def update(self) :
			dct = dict()
			for child in widgetsOf(self.layout) :
				dct.update(child.update())
			return dct


	class Entry(QtWidgets.QWidget) :

		def __init__(self, param, parent) :

			super().__init__(parent)

			self.parent = parent
			del parent
			self.param = param

			self.layout = QtWidgets.QHBoxLayout(self)

			self.label = QtWidgets.QLabel(self.param.name, self)
			if self.param.choices :
				self.selector = QtWidgets.QComboBox(self)
				self.selector.addItems(self.param.choices)
				self.selector.get = self.selector.currentText
				self.selector.set = self.selector.setCurrentText
			else :
				self.selector = QtWidgets.QLineEdit(self)
				self.selector.set = self.selector.setText
				self.selector.get = self.selector.text

			self.old = QtWidgets.QLabel(self.get() or 'None')

			self.layout.addWidget(self.label)
			self.layout.addWidget(self.old)
			self.layout.addWidget(self.selector)

		def get(self) :
			return self.selector.get()

		def update(self) :
			self.old.setText(self.get())
			self.selector.set(self.old.text())
			return self.old.text()

	def __init__(self, param, parent) :

		super().__init__(parent)

		self.parent = parent
		del parent
		self.param = param
		del param

		self.layout = QtWidgets.QVBoxLayout(self)
		self.entry = QNamedDict.Entry(self.param, self)
		self.entries = QNamedDict.Subentries(self.param, self)

		self.layout.addWidget(self.entry)
		self.layout.addWidget(self.entries)

	def get(self) :
		if self.param.children :
			#print(self.param)
			dct = {self.param.name : NamedDict(self.entry.get())}
			dct[self.param.name].update(self.entries.get())
		else :
			dct = {self.param.name:self.entry.get()}
		return dct

	def update(self) :
		if self.param.children :
			#print(self.param)
			dct = {self.param.name : NamedDict(self.entry.update())}
			dct[self.param.name].update(self.entries.update())
		else :
			dct = {self.param.name:self.entry.get()}
		return dct
