from PyQt5 import QtWidgets

import ptexture
import typing

class ptexture_params_gui(QtWidgets.QWidget) :

	def addGridWidget(self, widget : QtWidgets.QWidget,
	                        row : int = None,
													col : int = None) :
		# Row-major addition.
		if row == None :
			row = self.row
			if col == None :
				col = self.col
				self.col += 1

		self.grid.addWidget(widget, self.row_offset + row,
		                            self.col_offset + col)

	def nextGridRow(self) :

		self.row += 1
		self.col = 0


	def __init__(self, ptex : ptexture.ptexture,
	                   grid : QtWidgets.QGridLayout = None,
										 start : typing.Tuple[int, int] = None) :

		super().__init__()

		self.ptex = ptex
		del ptex

		if grid is None :
			grid = QtWidgets.QGridLayout(self)
			start = (0,0)
		self.grid = grid
		self.row_offset, self.col_offset = start
		self.row, self.col = 0,0
		del grid
		del start


		#self.mainLayout = QtWidgets.QVBoxLayout(self)

		self.dropdown = QtWidgets.QComboBox(self)
		self.dropdown.addItem(self.ptex.name)

		#self.mainLayout.addWidget(self.dropdown)
		self.addGridWidget(self.dropdown)
		self.nextGridRow()

		#entriesLayoutParent = QtWidgets.QWidget()
		#self.entriesLayout = QtWidgets.QVBoxLayout(entriesLayoutParent)
		self.inner_gui = dict()

		for param in self.ptex.params :
			if param.type == ptexture.ptexture :
				#self.entriesLayout.addWidget(ptexture_params_gui(param.type(param.default)))
				#entry = ptexture_dict_entry(param, QtWidgets.QComboBox, self,
				#QtWidgets.QComboBox.currentText, lambda dropdown :
				#dropdown.addItem(param.name))
				#self.entriesLayout.addWidget(entry)
				self.addGridWidget(QtWidgets.QLabel(param.name)) # name
				self.addGridWidget(QtWidgets.QLabel(str(param.default))) # current
				self.inner_gui[param.name] = \
					ptexture_params_gui(ptexture.ptexture(param.default), self.grid,
					(self.row, self.col_offset+2))
				self.row = self.grid.rowCount()
				self.col = self.col_offset
			else :
				#entry = ptexture_dict_entry(param)
				#self.entriesLayout.addWidget(entry)
				self.addGridWidget(QtWidgets.QLabel(param.name)) # name
				self.addGridWidget(QtWidgets.QLabel(str(param.default))) # current
				self.addGridWidget(QtWidgets.QLineEdit()) # new
				self.nextGridRow()



		#self.mainLayout.addWidget(entriesLayoutParent)

	def get(self) :

		import collections

		dct = collections.defaultdict(lambda:dict())
		ptex_name = self.dropdown.currentText()

		for r in range(self.grid.rowCount()) :
			name = self.grid.itemAtPosition(r, self.col_offset)
			print(r, self.col_offset, end=' ')
			if name is None  :
				print(name)
				continue
			else :
				name = name.widget()
				if not isinstance(name, QtWidgets.QLabel) :
					print()
					continue
				print(name.text())

			current = self.grid.itemAtPosition(r, self.col_offset+1).widget()
			new     = self.grid.itemAtPosition(r, self.col_offset+2).widget()

			if isinstance(new, QtWidgets.QComboBox) :
				inner_dct = self.inner_gui[name.text()].get()
				#dct.update(inner_dct)
				dct[ptex_name][name.text()] = inner_dct
				current.setText(new.currentText())
				continue

			if new.text() == '' :
				new.setText(current.text())

			dct[ptex_name][name.text()] = \
				ptexture.ptexture(ptex_name).params_dict[name.text()].type(new.text())

			current.setText(new.text())
			new.setText('')

		return dct





		#for i in range(self.entriesLayout.count()) :

		#	#entry = self.entriesLayout.itemAt(i).widget()

		#	if type(entry) == ptexture_params_gui :
		#		inner_ptex_name = entry.dropdown.currentText()
		#		inner_dct = entry.get()

		#		# doesn't handle repeats of a texture (when texture is passed as own
		#		# param)
		#		dct.update(inner_dct)

		#		continue

		#	name, current, new = \
		#		entry.name, entry.current, entry.new

		#	if new.text() == '' :
		#		new.setText(current.text())

		#	dct[ptex_name][name.text()] = \
		#		ptexture.ptexture(ptex_name).params_dict[name.text()].type(new.text())

		#	current.setText(new.text())
		#	new.setText('')

		#return dct

class ptexture_dict_entry(QtWidgets.QWidget) :

	def __init__(self,
		param : ptexture.typed_param,
		selector = QtWidgets.QLineEdit,
		args = tuple(),
		selector_get_text = QtWidgets.QLineEdit.text) :

		super().__init__()

		self.param = param
		del param

		self.name = QtWidgets.QLabel(self.param.name)
		self.current = QtWidgets.QLabel(str(self.param.default))
		self.new = selector(*args)

		self.layout = QtWidgets.QHBoxLayout(self)

		self.layout.addWidget(self.name)
		self.layout.addWidget(self.current)
		self.layout.addWidget(self.new)

