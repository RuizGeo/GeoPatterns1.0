# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'segmentation_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.resize(581, 259)
        Dialog.setWindowFlags(QtCore.Qt.WindowMinMaxButtonsHint)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMaximumSize(QtCore.QSize(581, 259))
        Dialog.setSizeGripEnabled(False)
        #Dialog.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(2, 9, 571, 241))
        self.tabWidget.setObjectName("tabWidget")
        self.tabInput = QtWidgets.QWidget()
        self.tabInput.setObjectName("tabInput")
        self.layoutWidget = QtWidgets.QWidget(self.tabInput)
        self.layoutWidget.setGeometry(QtCore.QRect(11, 11, 551, 50))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.labelIMG = QtWidgets.QLabel(self.layoutWidget)
        self.labelIMG.setObjectName("labelIMG")
        self.verticalLayout.addWidget(self.labelIMG)
        self.comboBoxRaster = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBoxRaster.setObjectName("comboBoxRaster")
        self.verticalLayout.addWidget(self.comboBoxRaster)
        self.tabWidget.addTab(self.tabInput, "")
        self.tabParameters = QtWidgets.QWidget()
        self.tabParameters.setObjectName("tabParameters")
        self.scrollArea = QtWidgets.QScrollArea(self.tabParameters)
        self.scrollArea.setGeometry(QtCore.QRect(9, 9, 551, 181))
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, -71, 535, 250))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.labelStartSim = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelStartSim.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStartSim.setObjectName("labelStartSim")
        self.gridLayout_3.addWidget(self.labelStartSim, 1, 0, 1, 1)
        self.spinBoxEndMcells = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxEndMcells.setMinimum(0)
        self.spinBoxEndMcells.setMaximum(100000)
        self.spinBoxEndMcells.setProperty("value", 200)
        self.spinBoxEndMcells.setObjectName("spinBoxEndMcells")
        self.gridLayout_3.addWidget(self.spinBoxEndMcells, 5, 3, 1, 1)
        self.spinBoxStepMcells = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxStepMcells.setMinimum(0)
        self.spinBoxStepMcells.setMaximum(100000)
        self.spinBoxStepMcells.setProperty("value", 100)
        self.spinBoxStepMcells.setObjectName("spinBoxStepMcells")
        self.gridLayout_3.addWidget(self.spinBoxStepMcells, 5, 5, 1, 1)
        self.labelEndSim = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelEndSim.setAlignment(QtCore.Qt.AlignCenter)
        self.labelEndSim.setObjectName("labelEndSim")
        self.gridLayout_3.addWidget(self.labelEndSim, 1, 2, 1, 1)
        self.labelSimMeth = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelSimMeth.setObjectName("labelSimMeth")
        self.gridLayout_3.addWidget(self.labelSimMeth, 6, 0, 1, 3)
        self.labelSimThres = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelSimThres.setObjectName("labelSimThres")
        self.gridLayout_3.addWidget(self.labelSimThres, 0, 0, 1, 5)
        self.labelStepMinCel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelStepMinCel.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStepMinCel.setObjectName("labelStepMinCel")
        self.gridLayout_3.addWidget(self.labelStepMinCel, 5, 4, 1, 1)
        self.labelStepSim = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelStepSim.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelStepSim.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStepSim.setObjectName("labelStepSim")
        self.gridLayout_3.addWidget(self.labelStepSim, 1, 4, 1, 1)
        self.spinBoxStarSim = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxStarSim.setDecimals(3)
        self.spinBoxStarSim.setMaximum(1.0)
        self.spinBoxStarSim.setSingleStep(0.001)
        self.spinBoxStarSim.setObjectName("spinBoxStarSim")
        self.gridLayout_3.addWidget(self.spinBoxStarSim, 1, 1, 1, 1)
        self.spinBoxStartMcells = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxStartMcells.setMinimum(0)
        self.spinBoxStartMcells.setMaximum(100000)
        self.spinBoxStartMcells.setObjectName("spinBoxStartMcells")
        self.gridLayout_3.addWidget(self.spinBoxStartMcells, 5, 1, 1, 1)
        self.labelAmountMem = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelAmountMem.setObjectName("labelAmountMem")
        self.gridLayout_3.addWidget(self.labelAmountMem, 8, 0, 1, 3)
        self.labelMaxIter = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelMaxIter.setObjectName("labelMaxIter")
        self.gridLayout_3.addWidget(self.labelMaxIter, 10, 0, 1, 3)
        self.spinBoxMiter = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxMiter.setMinimum(1)
        self.spinBoxMiter.setMaximum(100)
        self.spinBoxMiter.setProperty("value", 10)
        self.spinBoxMiter.setObjectName("spinBoxMiter")
        self.gridLayout_3.addWidget(self.spinBoxMiter, 10, 3, 1, 3)
        self.labelMaxCells = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelMaxCells.setObjectName("labelMaxCells")
        self.gridLayout_3.addWidget(self.labelMaxCells, 4, 0, 1, 6)
        self.spinBoxEndSim = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxEndSim.setDecimals(3)
        self.spinBoxEndSim.setMaximum(1.0)
        self.spinBoxEndSim.setSingleStep(0.001)
        self.spinBoxEndSim.setProperty("value", 1.0)
        self.spinBoxEndSim.setObjectName("spinBoxEndSim")
        self.gridLayout_3.addWidget(self.spinBoxEndSim, 1, 3, 1, 1)
        self.spinBoxStepSim = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxStepSim.setDecimals(3)
        self.spinBoxStepSim.setMaximum(1.0)
        self.spinBoxStepSim.setSingleStep(0.001)
        self.spinBoxStepSim.setProperty("value", 0.05)
        self.spinBoxStepSim.setObjectName("spinBoxStepSim")
        self.gridLayout_3.addWidget(self.spinBoxStepSim, 1, 5, 1, 1)
        self.labelEndMinCel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelEndMinCel.setAlignment(QtCore.Qt.AlignCenter)
        self.labelEndMinCel.setObjectName("labelEndMinCel")
        self.gridLayout_3.addWidget(self.labelEndMinCel, 5, 2, 1, 1)
        self.spinBoxMemory = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxMemory.setMinimum(100)
        self.spinBoxMemory.setMaximum(1000000)
        self.spinBoxMemory.setSingleStep(10)
        self.spinBoxMemory.setProperty("value", 300)
        self.spinBoxMemory.setObjectName("spinBoxMemory")
        self.gridLayout_3.addWidget(self.spinBoxMemory, 8, 3, 1, 3)
        self.comboBoxSimMeth = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.comboBoxSimMeth.setObjectName("comboBoxSimMeth")
        self.gridLayout_3.addWidget(self.comboBoxSimMeth, 6, 3, 1, 3)
        self.labelStartMinCel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelStartMinCel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelStartMinCel.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStartMinCel.setObjectName("labelStartMinCel")
        self.gridLayout_3.addWidget(self.labelStartMinCel, 5, 0, 1, 1)
        self.labelNeighbors = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelNeighbors.setObjectName("labelNeighbors")
        self.gridLayout_3.addWidget(self.labelNeighbors, 7, 0, 1, 3)
        self.comboBoxNeigh = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.comboBoxNeigh.setObjectName("comboBoxNeigh")
        self.gridLayout_3.addWidget(self.comboBoxNeigh, 7, 3, 1, 3)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.tabWidget.addTab(self.tabParameters, "")
        self.tabOutput = QtWidgets.QWidget()
        self.tabOutput.setObjectName("tabOutput")
        self.layoutWidget1 = QtWidgets.QWidget(self.tabOutput)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 10, 551, 213))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.buttonPathSegs = QtWidgets.QPushButton(self.layoutWidget1)
        self.buttonPathSegs.setObjectName("buttonPathSegs")
        self.gridLayout_2.addWidget(self.buttonPathSegs, 1, 1, 1, 1)
        self.lineEditPathSegs = QtWidgets.QLineEdit(self.layoutWidget1)
        self.lineEditPathSegs.setObjectName("lineEditPathSegs")
        self.gridLayout_2.addWidget(self.lineEditPathSegs, 1, 0, 1, 1)
        self.textEditOut = QtWidgets.QTextEdit(self.layoutWidget1)
        self.textEditOut.setEnabled(True)
        self.textEditOut.setObjectName("textEditOut")
        self.gridLayout_2.addWidget(self.textEditOut, 2, 0, 1, 2)
        self.labelAux = QtWidgets.QLabel(self.layoutWidget1)
        self.labelAux.setText("")
        self.labelAux.setObjectName("labelAux")
        self.gridLayout_2.addWidget(self.labelAux, 4, 0, 1, 2)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.buttonCancel = QtWidgets.QPushButton(self.layoutWidget1)
        self.buttonCancel.setObjectName("buttonCancel")
        self.gridLayout_5.addWidget(self.buttonCancel, 0, 0, 1, 1)
        self.buttonRun = QtWidgets.QPushButton(self.layoutWidget1)
        self.buttonRun.setObjectName("buttonRun")
        self.gridLayout_5.addWidget(self.buttonRun, 0, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem, 0, 1, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.layoutWidget1)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_5.addWidget(self.progressBar, 1, 0, 1, 3)
        self.gridLayout_2.addLayout(self.gridLayout_5, 3, 0, 1, 2)
        self.labelPathSegs = QtWidgets.QLabel(self.layoutWidget1)
        self.labelPathSegs.setObjectName("labelPathSegs")
        self.gridLayout_2.addWidget(self.labelPathSegs, 0, 0, 1, 2)
        self.tabWidget.addTab(self.tabOutput, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Segmentation - GRASSGIS"))
        self.labelIMG.setText(_translate("Dialog", "Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabInput), _translate("Dialog", "Input"))
        self.labelStartSim.setText(_translate("Dialog", "Start"))
        self.labelEndSim.setText(_translate("Dialog", "end"))
        self.labelSimMeth.setText(_translate("Dialog", "Similarity calculatiom method "))
        self.labelSimThres.setText(_translate("Dialog", "Similarity threshold "))
        self.labelStepMinCel.setText(_translate("Dialog", "step"))
        self.labelStepSim.setText(_translate("Dialog", "step"))
        self.labelAmountMem.setText(_translate("Dialog", "Amount of memory to use in MB"))
        self.labelMaxIter.setText(_translate("Dialog", "Maximum number of iterations"))
        self.labelMaxCells.setText(_translate("Dialog", "Minimum of cells in a number segment"))
        self.labelEndMinCel.setText(_translate("Dialog", "end"))
        self.labelStartMinCel.setText(_translate("Dialog", "Start"))
        self.labelNeighbors.setText(_translate("Dialog", "Neighbors"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabParameters), _translate("Dialog", "Parameters"))
        self.buttonPathSegs.setText(_translate("Dialog", "..."))
        self.buttonCancel.setText(_translate("Dialog", "Exit"))
        self.buttonRun.setText(_translate("Dialog", "Run"))
        self.labelPathSegs.setText(_translate("Dialog", "Segmentation path"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabOutput), _translate("Dialog", "Output"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

