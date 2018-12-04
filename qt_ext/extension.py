from PyQt5 import QtWidgets

def clearLayout(layout : QtWidgets.QLayout) :
	for i in reversed(range(layout.count())) :
		widget = layout.itemAt(i).widget()
		layout.removeWidget(widget)
		widget.setParent(None)
		widget.deleteLater()

def widgetsOf(layout : QtWidgets.QLayout) :
	return (layout.itemAt(i).widget() for i in range(layout.count()))
