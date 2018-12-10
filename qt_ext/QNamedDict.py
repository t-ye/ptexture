import dataclasses
import typing
from .extension import widgetsOf, clearLayout
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

		dct = dict()

class QNamedDict(QtWidgets.QWidget) :

		@staticmethod
		def createSelector(param, parent) :

			if param.choices :
				selector = QtWidgets.QComboBox(parent)
				selector.addItems(param.choices)
				selector.get = selector.currentText
				selector.handler = selector.currentTextChanged
			else :
				selector = QtWidgets.QLineEdit(parent)
				selector.get = selector.text
				selector.handler = selector.editingFinished

			return selector

		def __init__(self, param, parent) :

			super().__init__(parent)

			self.parent = parent
			self.param = param
			del parent
			del param

			self.createLayouts()
			self.populateHeaderLayout()
			self.attachSignals()

		def createLayouts(self) :

			self.mainLayout = QtWidgets.QVBoxLayout(self)
			self.headerLayoutFrame = QtWidgets.QFrame(self)
			self.headerLayout = QtWidgets.QHBoxLayout(self.headerLayoutFrame)

			self.subdictsLayoutFrame = QtWidgets.QFrame(self)
			self.subdictsLayout = QtWidgets.QVBoxLayout(self.subdictsLayoutFrame)

			self.mainLayout.addWidget(self.headerLayoutFrame)
			self.mainLayout.addWidget(self.subdictsLayoutFrame)

		def populateHeaderLayout(self) :

			self.label = QtWidgets.QLabel(self.param.name, self.headerLayoutFrame)
			self.selector = self.createSelector(self.param, self.headerLayoutFrame)

			self.headerLayout.addWidget(self.label)
			self.headerLayout.addWidget(self.selector)

		def clearSubdicts(self) :
			clearLayout(self.subdictsLayout)
			self.subdicts = []

		def addSubdict(self, subdict) :
				self.subdictsLayout.addWidget(subdict)
				self.subdicts.append(subdict)

		def updateSubdictsLayout(self) :
			self.clearSubdicts()

			params = self.param.get_params(self.get())
			for param in params :
				subdict = QNamedDict(param, self.subdictsLayoutFrame)
				self.addSubdict(subdict)

		def attachSignals(self) :
			self.selector.handler.connect(self.updateSubdictsLayout)
			self.updateSubdictsLayout()

		def get(self) :
			try :
				return self.param.parse(self.selector.get())
			except ValueError :
				return None

		def getDict(self) :

			if self.subdicts :
				nd = NamedDict(self.get())
				for subdict in self.subdicts :
					nd.update(subdict.getDict())
				dct = {self.param.name : nd}
			else :
				dct = {self.param.name : self.get()}

			return dct
