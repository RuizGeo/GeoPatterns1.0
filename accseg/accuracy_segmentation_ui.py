# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'accuracy_segmentation_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.resize(530, 260)
        Dialog.setWindowFlags(QtCore.Qt.WindowMinMaxButtonsHint)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMaximumSize(QtCore.QSize(530, 260))
        Dialog.setSizeGripEnabled(False)
        self.layoutWidget = QtWidgets.QWidget(Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 511, 241))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonTextAss = QtWidgets.QPushButton(self.layoutWidget)
        self.buttonTextAss.setObjectName("buttonTextAss")
        self.gridLayout.addWidget(self.buttonTextAss, 9, 3, 1, 1)
        self.buttonCancel = QtWidgets.QPushButton(self.layoutWidget)
        self.buttonCancel.setObjectName("buttonCancel")
        self.gridLayout.addWidget(self.buttonCancel, 11, 0, 1, 1)
        self.labelRefDat = QtWidgets.QLabel(self.layoutWidget)
        self.labelRefDat.setObjectName("labelRefDat")
        self.gridLayout.addWidget(self.labelRefDat, 0, 0, 1, 4)
        self.labelTextAss = QtWidgets.QLabel(self.layoutWidget)
        self.labelTextAss.setObjectName("labelTextAss")
        self.gridLayout.addWidget(self.labelTextAss, 8, 0, 1, 3)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 10, 0, 1, 4)
        self.labelSegs = QtWidgets.QLabel(self.layoutWidget)
        self.labelSegs.setObjectName("labelSegs")
        self.gridLayout.addWidget(self.labelSegs, 6, 0, 1, 3)
        self.buttonSegs = QtWidgets.QPushButton(self.layoutWidget)
        self.buttonSegs.setObjectName("buttonSegs")
        self.gridLayout.addWidget(self.buttonSegs, 7, 3, 1, 1)
        self.lineEditSegs = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEditSegs.setObjectName("lineEditSegs")
        self.gridLayout.addWidget(self.lineEditSegs, 7, 0, 1, 3)
        self.progressBar = QtWidgets.QProgressBar(self.layoutWidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 12, 0, 1, 4)
        self.buttonRun = QtWidgets.QPushButton(self.layoutWidget)
        self.buttonRun.setObjectName("buttonRun")
        self.gridLayout.addWidget(self.buttonRun, 11, 3, 1, 1)
        self.lineEditTextAss = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEditTextAss.setObjectName("lineEditTextAss")
        self.gridLayout.addWidget(self.lineEditTextAss, 9, 0, 1, 3)
        self.comboBoxRefDat = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBoxRefDat.setObjectName("comboBoxRefDat")
        self.gridLayout.addWidget(self.comboBoxRefDat, 3, 0, 1, 4)
        self.labelOut = QtWidgets.QLabel(self.layoutWidget)
        self.labelOut.setText("")
        self.labelOut.setObjectName("labelOut")
        self.labelOut.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.labelOut, 11, 1, 1, 2)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Accuracy of the segmentation"))
        self.buttonTextAss.setText(_translate("Dialog", "..."))
        self.buttonCancel.setText(_translate("Dialog", "Cancel"))
        self.labelRefDat.setText(_translate("Dialog", "Reference data (polygon)"))
        self.labelTextAss.setText(_translate("Dialog", "Text assessment "))
        self.labelSegs.setText(_translate("Dialog", "Segmentation (polygon)"))
        self.buttonSegs.setText(_translate("Dialog", "..."))
        self.buttonRun.setText(_translate("Dialog", "Run"))

