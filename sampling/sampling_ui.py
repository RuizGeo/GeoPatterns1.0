# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sampling_ui.ui'
#
# Created by: PyQt5 UI code generator 5.7
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
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 551, 181))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEditInRast = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEditInRast.setObjectName("lineEditInRast")
        self.gridLayout.addWidget(self.lineEditInRast, 1, 0, 1, 1)
        self.buttonInRast = QtWidgets.QPushButton(self.layoutWidget)
        self.buttonInRast.setObjectName("buttonInRast")
        self.gridLayout.addWidget(self.buttonInRast, 1, 1, 1, 1)
        self.labelRast = QtWidgets.QLabel(self.layoutWidget)
        self.labelRast.setObjectName("labelRast")
        self.gridLayout.addWidget(self.labelRast, 0, 0, 1, 1)
        self.buttonInVec = QtWidgets.QPushButton(self.layoutWidget)
        self.buttonInVec.setObjectName("buttonInVec")
        self.gridLayout.addWidget(self.buttonInVec, 3, 1, 1, 1)
        self.lineEditVec = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEditVec.setObjectName("lineEditVec")
        self.gridLayout.addWidget(self.lineEditVec, 3, 0, 1, 1)
        self.labelVec = QtWidgets.QLabel(self.layoutWidget)
        self.labelVec.setObjectName("labelVec")
        self.gridLayout.addWidget(self.labelVec, 2, 0, 1, 1)
        self.tabWidget.addTab(self.tabInput, "")
        self.tabParar = QtWidgets.QWidget()
        self.tabParar.setObjectName("tabParar")
        self.groupBox = QtWidgets.QGroupBox(self.tabParar)
        self.groupBox.setGeometry(QtCore.QRect(20, 10, 531, 191))
        self.groupBox.setObjectName("groupBox")
        self.widget = QtWidgets.QWidget(self.groupBox)
        self.widget.setGeometry(QtCore.QRect(20, 30, 491, 151))
        self.widget.setObjectName("widget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.checkBoxSum = QtWidgets.QCheckBox(self.widget)
        self.checkBoxSum.setObjectName("checkBoxSum")
        self.gridLayout_3.addWidget(self.checkBoxSum, 0, 0, 1, 1)
        self.checkBoxSTD = QtWidgets.QCheckBox(self.widget)
        self.checkBoxSTD.setChecked(True)
        self.checkBoxSTD.setObjectName("checkBoxSTD")
        self.gridLayout_3.addWidget(self.checkBoxSTD, 0, 1, 1, 1)
        self.checkBoxMean = QtWidgets.QCheckBox(self.widget)
        self.checkBoxMean.setChecked(True)
        self.checkBoxMean.setObjectName("checkBoxMean")
        self.gridLayout_3.addWidget(self.checkBoxMean, 1, 0, 1, 1)
        self.checkBoxRange = QtWidgets.QCheckBox(self.widget)
        self.checkBoxRange.setObjectName("checkBoxRange")
        self.gridLayout_3.addWidget(self.checkBoxRange, 1, 1, 1, 1)
        self.checkBoxMedian = QtWidgets.QCheckBox(self.widget)
        self.checkBoxMedian.setObjectName("checkBoxMedian")
        self.gridLayout_3.addWidget(self.checkBoxMedian, 2, 0, 1, 1)
        self.checkBoxMinor = QtWidgets.QCheckBox(self.widget)
        self.checkBoxMinor.setObjectName("checkBoxMinor")
        self.gridLayout_3.addWidget(self.checkBoxMinor, 2, 1, 1, 1)
        self.checkBoxMin = QtWidgets.QCheckBox(self.widget)
        self.checkBoxMin.setObjectName("checkBoxMin")
        self.gridLayout_3.addWidget(self.checkBoxMin, 3, 0, 1, 1)
        self.checkBoxMajor = QtWidgets.QCheckBox(self.widget)
        self.checkBoxMajor.setObjectName("checkBoxMajor")
        self.gridLayout_3.addWidget(self.checkBoxMajor, 3, 1, 1, 1)
        self.checkBoxMax = QtWidgets.QCheckBox(self.widget)
        self.checkBoxMax.setObjectName("checkBoxMax")
        self.gridLayout_3.addWidget(self.checkBoxMax, 4, 0, 1, 1)
        self.checkBoxVariety = QtWidgets.QCheckBox(self.widget)
        self.checkBoxVariety.setObjectName("checkBoxVariety")
        self.gridLayout_3.addWidget(self.checkBoxVariety, 4, 1, 1, 1)
        self.tabWidget.addTab(self.tabParar, "")
        self.tabOutput = QtWidgets.QWidget()
        self.tabOutput.setObjectName("tabOutput")
        self.layoutWidget1 = QtWidgets.QWidget(self.tabOutput)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 10, 551, 213))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.textEditOut = QtWidgets.QTextEdit(self.layoutWidget1)
        self.textEditOut.setEnabled(True)
        self.textEditOut.setObjectName("textEditOut")
        self.gridLayout_2.addWidget(self.textEditOut, 0, 0, 1, 2)
        self.labelAux = QtWidgets.QLabel(self.layoutWidget1)
        self.labelAux.setText("")
        self.labelAux.setObjectName("labelAux")
        self.gridLayout_2.addWidget(self.labelAux, 2, 0, 1, 2)
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
        self.gridLayout_2.addLayout(self.gridLayout_5, 1, 0, 1, 2)
        self.tabWidget.addTab(self.tabOutput, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Sampling - Zonal Statistics"))
        self.buttonInRast.setText(_translate("Dialog", "..."))
        self.labelRast.setText(_translate("Dialog", "Features path (Rasters)"))
        self.buttonInVec.setText(_translate("Dialog", "..."))
        self.labelVec.setText(_translate("Dialog", "Segmentations path (vectors)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabInput), _translate("Dialog", "Input"))
        self.groupBox.setTitle(_translate("Dialog", "Statistics"))
        self.checkBoxSum.setText(_translate("Dialog", "Sum"))
        self.checkBoxSTD.setText(_translate("Dialog", "Standard deviation"))
        self.checkBoxMean.setText(_translate("Dialog", "Mean"))
        self.checkBoxRange.setText(_translate("Dialog", " Range  (max - min)"))
        self.checkBoxMedian.setText(_translate("Dialog", "Median"))
        self.checkBoxMinor.setText(_translate("Dialog", "Minority"))
        self.checkBoxMin.setText(_translate("Dialog", " Minimum"))
        self.checkBoxMajor.setText(_translate("Dialog", "Majority"))
        self.checkBoxMax.setText(_translate("Dialog", " Maximum"))
        self.checkBoxVariety.setText(_translate("Dialog", "Variety"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabParar), _translate("Dialog", "Parameter"))
        self.buttonCancel.setText(_translate("Dialog", "Cancel"))
        self.buttonRun.setText(_translate("Dialog", "Run"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabOutput), _translate("Dialog", "Output"))

