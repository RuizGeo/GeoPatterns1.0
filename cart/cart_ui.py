# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cart_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.resize(538, 259)
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
        self.tabWidget.setGeometry(QtCore.QRect(2, 9, 531, 241))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")
        self.tabInput = QtWidgets.QWidget()
        self.tabInput.setObjectName("tabInput")
        self.layoutWidget = QtWidgets.QWidget(self.tabInput)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 511, 195))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.comboBoxVal = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBoxVal.setObjectName("comboBoxVal")
        self.gridLayout.addWidget(self.comboBoxVal, 7, 0, 1, 1)
        self.labelClassVal = QtWidgets.QLabel(self.layoutWidget)
        self.labelClassVal.setAlignment(QtCore.Qt.AlignCenter)
        self.labelClassVal.setObjectName("labelClassVal")
        self.gridLayout.addWidget(self.labelClassVal, 6, 1, 1, 2)
        self.labelVal = QtWidgets.QLabel(self.layoutWidget)
        self.labelVal.setObjectName("labelVal")
        self.gridLayout.addWidget(self.labelVal, 6, 0, 1, 1)
        self.lineEditDataSet = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEditDataSet.setObjectName("lineEditDataSet")
        self.gridLayout.addWidget(self.lineEditDataSet, 9, 0, 1, 1)
        self.buttonDataSet = QtWidgets.QPushButton(self.layoutWidget)
        self.buttonDataSet.setObjectName("buttonDataSet")
        self.gridLayout.addWidget(self.buttonDataSet, 9, 1, 1, 2)
        self.comboBoxTrain = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBoxTrain.setObjectName("comboBoxTrain")
        self.gridLayout.addWidget(self.comboBoxTrain, 3, 0, 1, 1)
        self.labelClassTrain = QtWidgets.QLabel(self.layoutWidget)
        self.labelClassTrain.setEnabled(True)
        self.labelClassTrain.setSizeIncrement(QtCore.QSize(100, 0))
        self.labelClassTrain.setAlignment(QtCore.Qt.AlignCenter)
        self.labelClassTrain.setObjectName("labelClassTrain")
        self.gridLayout.addWidget(self.labelClassTrain, 0, 1, 1, 2)
        self.labelTrain = QtWidgets.QLabel(self.layoutWidget)
        self.labelTrain.setObjectName("labelTrain")
        self.gridLayout.addWidget(self.labelTrain, 0, 0, 1, 1)
        self.comboBoxFieldVal = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBoxFieldVal.setObjectName("comboBoxFieldVal")
        self.gridLayout.addWidget(self.comboBoxFieldVal, 7, 1, 1, 2)
        self.comboBoxFieldTrain = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBoxFieldTrain.setObjectName("comboBoxFieldTrain")
        self.gridLayout.addWidget(self.comboBoxFieldTrain, 3, 1, 1, 2)
        self.labelDataSet = QtWidgets.QLabel(self.layoutWidget)
        self.labelDataSet.setObjectName("labelDataSet")
        self.gridLayout.addWidget(self.labelDataSet, 8, 0, 1, 1)
        self.lineEditPackPath = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEditPackPath.setEnabled(False)
        self.lineEditPackPath.setObjectName("lineEditPackPath")
        self.gridLayout.addWidget(self.lineEditPackPath, 10, 0, 1, 1)
        self.checkBoxPackPath = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkBoxPackPath.setObjectName("checkBoxPackPath")
        self.gridLayout.addWidget(self.checkBoxPackPath, 10, 1, 1, 2)
        self.tabWidget.addTab(self.tabInput, "")
        self.tabParameters = QtWidgets.QWidget()
        self.tabParameters.setObjectName("tabParameters")
        self.scrollArea = QtWidgets.QScrollArea(self.tabParameters)
        self.scrollArea.setGeometry(QtCore.QRect(9, 9, 511, 181))
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 495, 213))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.spinBoxStepMinSam = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxStepMinSam.setMinimum(5)
        self.spinBoxStepMinSam.setMaximum(100)
        self.spinBoxStepMinSam.setSingleStep(5)
        self.spinBoxStepMinSam.setProperty("value", 5)
        self.spinBoxStepMinSam.setObjectName("spinBoxStepMinSam")
        self.gridLayout_3.addWidget(self.spinBoxStepMinSam, 3, 5, 1, 1)
        self.spinBoxEndMinSam = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxEndMinSam.setMinimum(5)
        self.spinBoxEndMinSam.setMaximum(100)
        self.spinBoxEndMinSam.setProperty("value", 30)
        self.spinBoxEndMinSam.setObjectName("spinBoxEndMinSam")
        self.gridLayout_3.addWidget(self.spinBoxEndMinSam, 3, 3, 1, 1)
        self.labelStepMinSam = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelStepMinSam.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelStepMinSam.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStepMinSam.setObjectName("labelStepMinSam")
        self.gridLayout_3.addWidget(self.labelStepMinSam, 3, 4, 1, 1)
        self.radioButtonRegress = QtWidgets.QRadioButton(self.scrollAreaWidgetContents)
        self.radioButtonRegress.setObjectName("radioButtonRegress")
        self.gridLayout_3.addWidget(self.radioButtonRegress, 0, 3, 1, 3)
        self.labelDepth = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelDepth.setObjectName("labelDepth")
        self.gridLayout_3.addWidget(self.labelDepth, 6, 0, 1, 6)
        self.labelMinSam = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelMinSam.setObjectName("labelMinSam")
        self.gridLayout_3.addWidget(self.labelMinSam, 2, 0, 1, 5)
        self.labelStartMinSam = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelStartMinSam.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStartMinSam.setObjectName("labelStartMinSam")
        self.gridLayout_3.addWidget(self.labelStartMinSam, 3, 0, 1, 1)
        self.spinBoxStartDepth = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxStartDepth.setMinimum(5)
        self.spinBoxStartDepth.setMaximum(100)
        self.spinBoxStartDepth.setObjectName("spinBoxStartDepth")
        self.gridLayout_3.addWidget(self.spinBoxStartDepth, 7, 1, 1, 1)
        self.spinBoxStartMinSam = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxStartMinSam.setMinimum(5)
        self.spinBoxStartMinSam.setMaximum(100)
        self.spinBoxStartMinSam.setObjectName("spinBoxStartMinSam")
        self.gridLayout_3.addWidget(self.spinBoxStartMinSam, 3, 1, 1, 1)
        self.labelEndDepth = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelEndDepth.setAlignment(QtCore.Qt.AlignCenter)
        self.labelEndDepth.setObjectName("labelEndDepth")
        self.gridLayout_3.addWidget(self.labelEndDepth, 7, 2, 1, 1)
        self.labelEndMinSam = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelEndMinSam.setAlignment(QtCore.Qt.AlignCenter)
        self.labelEndMinSam.setObjectName("labelEndMinSam")
        self.gridLayout_3.addWidget(self.labelEndMinSam, 3, 2, 1, 1)
        self.labelStartDepth = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelStartDepth.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelStartDepth.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStartDepth.setObjectName("labelStartDepth")
        self.gridLayout_3.addWidget(self.labelStartDepth, 7, 0, 1, 1)
        self.labelStepDepth = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelStepDepth.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStepDepth.setObjectName("labelStepDepth")
        self.gridLayout_3.addWidget(self.labelStepDepth, 7, 4, 1, 1)
        self.spinBoxEndDepth = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxEndDepth.setMinimum(5)
        self.spinBoxEndDepth.setMaximum(100)
        self.spinBoxEndDepth.setProperty("value", 30)
        self.spinBoxEndDepth.setObjectName("spinBoxEndDepth")
        self.gridLayout_3.addWidget(self.spinBoxEndDepth, 7, 3, 1, 1)
        self.spinBoxStepDepth = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxStepDepth.setMinimum(5)
        self.spinBoxStepDepth.setMaximum(100)
        self.spinBoxStepDepth.setSingleStep(5)
        self.spinBoxStepDepth.setProperty("value", 5)
        self.spinBoxStepDepth.setObjectName("spinBoxStepDepth")
        self.gridLayout_3.addWidget(self.spinBoxStepDepth, 7, 5, 1, 1)
        self.checkBoxApplyModel = QtWidgets.QCheckBox(self.scrollAreaWidgetContents)
        self.checkBoxApplyModel.setObjectName("checkBoxApplyModel")
        self.gridLayout_3.addWidget(self.checkBoxApplyModel, 9, 0, 1, 3)
        self.radioButtonClass = QtWidgets.QRadioButton(self.scrollAreaWidgetContents)
        self.radioButtonClass.setObjectName("radioButtonClass")
        self.gridLayout_3.addWidget(self.radioButtonClass, 0, 0, 1, 3)
        self.comboBoxCrit = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.comboBoxCrit.setObjectName("comboBoxCrit")
        self.gridLayout_3.addWidget(self.comboBoxCrit, 8, 4, 1, 2)
        self.labelCrit = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelCrit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelCrit.setObjectName("labelCrit")
        self.gridLayout_3.addWidget(self.labelCrit, 8, 0, 1, 4)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.tabWidget.addTab(self.tabParameters, "")
        self.tabOutput = QtWidgets.QWidget()
        self.tabOutput.setObjectName("tabOutput")
        self.layoutWidget1 = QtWidgets.QWidget(self.tabOutput)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 10, 511, 213))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.buttonAssessFile = QtWidgets.QPushButton(self.layoutWidget1)
        self.buttonAssessFile.setObjectName("buttonAssessFile")
        self.gridLayout_2.addWidget(self.buttonAssessFile, 1, 1, 1, 1)
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
        self.gridLayout_2.addLayout(self.gridLayout_5, 4, 0, 1, 2)
        self.labelRFC = QtWidgets.QLabel(self.layoutWidget1)
        self.labelRFC.setObjectName("labelRFC")
        self.gridLayout_2.addWidget(self.labelRFC, 0, 0, 1, 2)
        self.labelAux = QtWidgets.QLabel(self.layoutWidget1)
        self.labelAux.setText("")
        self.labelAux.setObjectName("labelAux")
        self.gridLayout_2.addWidget(self.labelAux, 5, 0, 1, 2)
        self.labelOutClass = QtWidgets.QLabel(self.layoutWidget1)
        self.labelOutClass.setObjectName("labelOutClass")
        self.gridLayout_2.addWidget(self.labelOutClass, 2, 0, 1, 1)
        self.lineEditAssessFile = QtWidgets.QLineEdit(self.layoutWidget1)
        self.lineEditAssessFile.setObjectName("lineEditAssessFile")
        self.gridLayout_2.addWidget(self.lineEditAssessFile, 1, 0, 1, 1)
        self.lineEditOutModel = QtWidgets.QLineEdit(self.layoutWidget1)
        self.lineEditOutModel.setObjectName("lineEditOutModel")
        self.gridLayout_2.addWidget(self.lineEditOutModel, 3, 0, 1, 1)
        self.buttonOutModel = QtWidgets.QPushButton(self.layoutWidget1)
        self.buttonOutModel.setObjectName("buttonOutModel")
        self.gridLayout_2.addWidget(self.buttonOutModel, 3, 1, 1, 1)
        self.tabWidget.addTab(self.tabOutput, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "CART"))
        self.labelClassVal.setText(_translate("Dialog", "target field"))
        self.labelVal.setText(_translate("Dialog", "Validation samples (Points)"))
        self.buttonDataSet.setText(_translate("Dialog", "..."))
        self.labelClassTrain.setText(_translate("Dialog", "target field"))
        self.labelTrain.setText(_translate("Dialog", "Training samples (Points)"))
        self.labelDataSet.setText(_translate("Dialog", "Data set"))
        self.checkBoxPackPath.setText(_translate("Dialog", "Packages"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabInput), _translate("Dialog", "Input"))
        self.labelStepMinSam.setText(_translate("Dialog", "step"))
        self.radioButtonRegress.setText(_translate("Dialog", "Regression"))
        self.labelDepth.setText(_translate("Dialog", "The maximum depth of the tree"))
        self.labelMinSam.setText(_translate("Dialog", "The minimum number of samples split an internal node"))
        self.labelStartMinSam.setText(_translate("Dialog", "start"))
        self.labelEndDepth.setText(_translate("Dialog", "end"))
        self.labelEndMinSam.setText(_translate("Dialog", "end"))
        self.labelStartDepth.setText(_translate("Dialog", "start"))
        self.labelStepDepth.setText(_translate("Dialog", "step"))
        self.checkBoxApplyModel.setText(_translate("Dialog", "Apply model"))
        self.radioButtonClass.setText(_translate("Dialog", "Classification"))
        self.labelCrit.setText(_translate("Dialog", "The function to measure the quality of a split "))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabParameters), _translate("Dialog", "Parameters"))
        self.buttonAssessFile.setText(_translate("Dialog", "..."))
        self.buttonCancel.setText(_translate("Dialog", "Exit"))
        self.buttonRun.setText(_translate("Dialog", "Run"))
        self.labelRFC.setText(_translate("Dialog", "Text file"))
        self.labelOutClass.setText(_translate("Dialog", "Vector file"))
        self.buttonOutModel.setText(_translate("Dialog", "..."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabOutput), _translate("Dialog", "Output"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

