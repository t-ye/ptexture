import dataclasses
import typing
from .extension import widgetsOf
from PyQt5 import QtWidgets
import functional

class NamedDict(dict) :
	"""
	Dictionary with a name attribute.
	"""

	def __init__(self, name, *args, **kwargs) :

		super().__init__(*args, **kwargs)
		self.name = name

	def __repr__(self) :
		return f'NamedDict(name={self.name}, {super().__repr__()})'

class QNamedDict(QtWidgets.QWidget) :
	"""
	GUI for inputting nested dictionaries.

	An example of a dictionary's format:

	root = a :
  			 b = c
	  		 d = e :
					   f = g
						 h = i
				 j = k

	Here, root and d are keys into the NamedDict's with names a and e,
	respectively.

	"""

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

		class Selector(QtWidgets.QWidget) :

			def __init__(self, param, parent) :

				super().__init__()

				self.parent = parent
				del parent
				self.param = param
				del param

				self.layout = QtWidgets.QHBoxLayout(self)
				self.layout.setContentsMargins(0,0,0,0)
				#self.layout.setSpacing(0)

				if self.param.choices :
					self.selector = QtWidgets.QComboBox(self)
					self.selector.addItems(self.param.choices)
					self.selector.get = self.selector.currentText
					self.selector.set = self.selector.setCurrentText
				else :
					self.selector = QtWidgets.QLineEdit(self)
					self.selector.set = self.selector.setText
					self.selector.get = self.selector.text

				self.layout.addWidget(self.selector)

			def get(self) :
				return self.selector.get()

		def __init__(self, param, parent) :

			super().__init__(parent)

			self.parent = parent
			del parent
			self.param = param

			self.layout = QtWidgets.QHBoxLayout(self)
			self.layout.setContentsMargins(0,0,0,0)

			self.label = QtWidgets.QLabel(self.param.name, self)
			self.selector = QNamedDict.Entry.Selector(param, self)
			self.old = QtWidgets.QLabel(self.selector.get())

			self.layout.addWidget(self.label)
			self.layout.addWidget(self.old)
			self.layout.addWidget(self.selector)

		def get(self) :
			return self.old.text()

		def update(self) :
			if self.selector.get() :
				self.old.setText(self.selector.get())
				# self.selector.set(self.old.text())
			return self.get()

	def __init__(self, param, parent) :

		super().__init__(parent)

		self.parent = parent
		del parent
		self.param = param
		del param

		self.layout = QtWidgets.QVBoxLayout(self)
		margins = self.layout.getContentsMargins()
		self.layout.setContentsMargins(margins[0], 0, 0, margins[3])

		self.entry = QNamedDict.Entry(self.param, self)
		self.entries = QNamedDict.Subentries(self.entry, self)

		self.layout.addWidget(self.entry)
		self.layout.addWidget(self.entries)

	def get(self) :
		"""
		Get old dictionary, and do not update children.
		"""
		if self.param.children :
			dct = {self.param.name : NamedDict(self.entry.get())}
			dct[self.param.name].update(self.entries.get())
		else :
			dct = {self.param.name:self.entry.get()}
		return dct

	def update(self) :
		"""
		Update children and get resultant dictionary.
		"""
		if self.param.children :
			dct = {self.param.name : NamedDict(self.entry.update())}
			dct[self.param.name].update(self.entries.update())
		else :
			dct = {self.param.name:self.entry.update()}
		return dct
