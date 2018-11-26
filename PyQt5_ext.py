from PyQt5 import QtWidgets

import ptexture


class ptexture_params_gui(QtWidgets.QWidget) :

	def __init__(self, ptex : ptexture.ptexture) :

		super().__init__()

		self.ptex = ptex
		del ptex

		self.mainLayout = QtWidgets.QVBoxLayout(self)

		self.dropdown = QtWidgets.QComboBox(self)
		self.dropdown.addItem(self.ptex.name)

		self.mainLayout.addWidget(self.dropdown)

		for param in self.ptex.params :
			entry = ptexture_dict_entry(param)
			self.mainLayout.addWidget(entry)

			if param.type == ptexture.ptexture :
				self.mainLayout.addWidget(ptexture_params_gui(param.type(param.default)))

	def get(self) :

		import collections

		dct = collections.defaultdict(lambda:dict())
		ptex_name = self.dropdown.currentText()

		for i in range(1,self.mainLayout.count()) :

			entry = self.mainLayout.itemAt(i).widget()


			if type(entry) == ptexture_params_gui :
				inner_ptex_name = entry.dropdown.currentText()
				inner_dct = entry.get()

				# doesn't handle repeats of a texture (when texture is passed as own
				# param)
				dct.update(inner_dct)

				continue

			name, current, new = \
				entry.name, entry.current, entry.new

			if new.text() == '' :
				new.setText(current.text())

			dct[ptex_name][name.text()] = \
				ptexture.ptexture(ptex_name).params_dict[name.text()].type(new.text())

			current.setText(new.text())
			new.setText('')

		return dct

class ptexture_dict_entry(QtWidgets.QWidget) :

	def __init__(self, param : ptexture.typed_param) :

		super().__init__()

		self.param = param
		del param

		self.name = QtWidgets.QLabel(self.param.name)
		self.current = QtWidgets.QLabel(str(self.param.default))
		self.new = QtWidgets.QLineEdit()

		self.layout = QtWidgets.QHBoxLayout(self)

		self.layout.addWidget(self.name)
		self.layout.addWidget(self.current)
		self.layout.addWidget(self.new)

