# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'geometrics_ui.ui'
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
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(2, 9, 571, 241))
        self.tabWidget.setObjectName("tabWidget")
        self.tabInput = QtWidgets.QWidget()
        self.tabInput.setObjectName("tabInput")
        self.layoutWidget = QtWidgets.QWidget(self.tabInput)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 551, 91))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonInVec = QtWidgets.QPushButton(self.layoutWidget)
        self.buttonInVec.setObjectName("buttonInVec")
        self.gridLayout.addWidget(self.buttonInVec, 1, 1, 1, 1)
        self.lineEditVec = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEditVec.setObjectName("lineEditVec")
        self.gridLayout.addWidget(self.lineEditVec, 1, 0, 1, 1)
        self.labelVec = QtWidgets.QLabel(self.layoutWidget)
        self.labelVec.setObjectName("labelVec")
        self.gridLayout.addWidget(self.labelVec, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tabInput, "")
        self.tabParar = QtWidgets.QWidget()
        self.tabParar.setObjectName("tabParar")
        self.groupBox = QtWidgets.QGroupBox(self.tabParar)
        self.groupBox.setGeometry(QtCore.QRect(20, 10, 531, 191))
        self.groupBox.setObjectName("groupBox")
        self.layoutWidget1 = QtWidgets.QWidget(self.groupBox)
        self.layoutWidget1.setGeometry(QtCore.QRect(20, 30, 491, 151))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.checkBoxArea = QtWidgets.QCheckBox(self.layoutWidget1)
        self.checkBoxArea.setObjectName("checkBoxArea")
        self.gridLayout_3.addWidget(self.checkBoxArea, 0, 0, 1, 1)
        self.checkBoxRoud = QtWidgets.QCheckBox(self.layoutWidget1)
        self.checkBoxRoud.setChecked(True)
        self.checkBoxRoud.setObjectName("checkBoxRoud")
        self.gridLayout_3.addWidget(self.checkBoxRoud, 0, 1, 1, 1)
        self.checkBoxPerimet = QtWidgets.QCheckBox(self.layoutWidget1)
        self.checkBoxPerimet.setChecked(True)
        self.checkBoxPerimet.setObjectName("checkBoxPerimet")
        self.gridLayout_3.addWidget(self.checkBoxPerimet, 1, 0, 1, 1)
        self.checkBoxFatCirc = QtWidgets.QCheckBox(self.layoutWidget1)
        self.checkBoxFatCirc.setObjectName("checkBoxFatCirc")
        self.gridLayout_3.addWidget(self.checkBoxFatCirc, 1, 1, 1, 1)
        self.checkBoxCompact = QtWidgets.QCheckBox(self.layoutWidget1)
        self.checkBoxCompact.setObjectName("checkBoxCompact")
        self.gridLayout_3.addWidget(self.checkBoxCompact, 2, 0, 1, 1)
        self.checkBoxSoft = QtWidgets.QCheckBox(self.layoutWidget1)
        self.checkBoxSoft.setObjectName("checkBoxSoft")
        self.gridLayout_3.addWidget(self.checkBoxSoft, 3, 0, 1, 1)
        self.checkBoxRet = QtWidgets.QCheckBox(self.layoutWidget1)
        self.checkBoxRet.setObjectName("checkBoxRet")
        self.gridLayout_3.addWidget(self.checkBoxRet, 2, 1, 1, 1)
        self.tabWidget.addTab(self.tabParar, "")
        self.tabOutput = QtWidgets.QWidget()
        self.tabOutput.setObjectName("tabOutput")
        self.layoutWidget2 = QtWidgets.QWidget(self.tabOutput)
        self.layoutWidget2.setGeometry(QtCore.QRect(10, 10, 551, 213))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.textEditOut = QtWidgets.QTextEdit(self.layoutWidget2)
        self.textEditOut.setEnabled(False)
        self.textEditOut.setObjectName("textEditOut")
        self.gridLayout_2.addWidget(self.textEditOut, 0, 0, 1, 2)
        self.labelAux = QtWidgets.QLabel(self.layoutWidget2)
        self.labelAux.setText("")
        self.labelAux.setObjectName("labelAux")
        self.gridLayout_2.addWidget(self.labelAux, 2, 0, 1, 2)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.buttonExit = QtWidgets.QPushButton(self.layoutWidget2)
        self.buttonExit.setObjectName("buttonExit")
        self.gridLayout_5.addWidget(self.buttonExit, 0, 0, 1, 1)
        self.buttonRun = QtWidgets.QPushButton(self.layoutWidget2)
        self.buttonRun.setObjectName("buttonRun")
        self.gridLayout_5.addWidget(self.buttonRun, 0, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem, 0, 1, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.layoutWidget2)
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
        Dialog.setWindowTitle(_translate("Dialog", "Geometric features"))
        self.buttonInVec.setText(_translate("Dialog", "..."))
        self.labelVec.setText(_translate("Dialog", "Vectors (Polygon)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabInput), _translate("Dialog", "Input"))
        self.groupBox.setTitle(_translate("Dialog", "Geometric features"))
        self.checkBoxArea.setText(_translate("Dialog", "Area (hectare)"))
        self.checkBoxRoud.setText(_translate("Dialog", "Roundness"))
        self.checkBoxPerimet.setText(_translate("Dialog", "Perimeter (meter)"))
        self.checkBoxFatCirc.setText(_translate("Dialog", "Factor form circular"))
        self.checkBoxCompact.setText(_translate("Dialog", "Compactness"))
        self.checkBoxSoft.setText(_translate("Dialog", "Softness"))
        self.checkBoxRet.setText(_translate("Dialog", " Rectangularity"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabParar), _translate("Dialog", "Parameter"))
        self.buttonExit.setText(_translate("Dialog", "Exit"))
        self.buttonRun.setText(_translate("Dialog", "Run"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabOutput), _translate("Dialog", "Output"))

