# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KNN_ui.ui'
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
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 551, 218))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonDataSetPath = QtWidgets.QPushButton(self.layoutWidget)
        self.buttonDataSetPath.setObjectName("buttonDataSetPath")
        self.gridLayout.addWidget(self.buttonDataSetPath, 9, 1, 1, 1)
        self.labelTrain = QtWidgets.QLabel(self.layoutWidget)
        self.labelTrain.setObjectName("labelTrain")
        self.gridLayout.addWidget(self.labelTrain, 0, 0, 1, 1)
        self.labelVal = QtWidgets.QLabel(self.layoutWidget)
        self.labelVal.setObjectName("labelVal")
        self.gridLayout.addWidget(self.labelVal, 6, 0, 1, 1)
        self.comboBoxClassVal = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBoxClassVal.setObjectName("comboBoxClassVal")
        self.gridLayout.addWidget(self.comboBoxClassVal, 7, 1, 1, 1)
        self.labelClassTrain = QtWidgets.QLabel(self.layoutWidget)
        self.labelClassTrain.setObjectName("labelClassTrain")
        self.gridLayout.addWidget(self.labelClassTrain, 0, 1, 1, 1)
        self.labelClassVal = QtWidgets.QLabel(self.layoutWidget)
        self.labelClassVal.setObjectName("labelClassVal")
        self.gridLayout.addWidget(self.labelClassVal, 6, 1, 1, 1)
        self.lineEditDataSetPath = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEditDataSetPath.setObjectName("lineEditDataSetPath")
        self.gridLayout.addWidget(self.lineEditDataSetPath, 9, 0, 1, 1)
        self.comboBoxVal = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBoxVal.setObjectName("comboBoxVal")
        self.gridLayout.addWidget(self.comboBoxVal, 7, 0, 1, 1)
        self.labelDataSet = QtWidgets.QLabel(self.layoutWidget)
        self.labelDataSet.setObjectName("labelDataSet")
        self.gridLayout.addWidget(self.labelDataSet, 8, 0, 1, 1)
        self.comboBoxTrain = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBoxTrain.setObjectName("comboBoxTrain")
        self.gridLayout.addWidget(self.comboBoxTrain, 3, 0, 1, 1)
        self.comboBoxClassTrain = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBoxClassTrain.setObjectName("comboBoxClassTrain")
        self.gridLayout.addWidget(self.comboBoxClassTrain, 3, 1, 1, 1)
        self.checkBoxPackPath_2 = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkBoxPackPath_2.setObjectName("checkBoxPackPath_2")
        self.gridLayout.addWidget(self.checkBoxPackPath_2, 11, 0, 1, 1)
        self.lineEditPackPath = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEditPackPath.setEnabled(False)
        self.lineEditPackPath.setObjectName("lineEditPackPath")
        self.gridLayout.addWidget(self.lineEditPackPath, 10, 0, 1, 1)
        self.checkBoxPackPath = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkBoxPackPath.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.checkBoxPackPath.setObjectName("checkBoxPackPath")
        self.gridLayout.addWidget(self.checkBoxPackPath, 10, 1, 1, 1)
        self.tabWidget.addTab(self.tabInput, "")
        self.tabParameters = QtWidgets.QWidget()
        self.tabParameters.setObjectName("tabParameters")
        self.scrollArea = QtWidgets.QScrollArea(self.tabParameters)
        self.scrollArea.setGeometry(QtCore.QRect(9, 9, 551, 181))
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 535, 185))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.labelWeights = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelWeights.setObjectName("labelWeights")
        self.gridLayout_3.addWidget(self.labelWeights, 5, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_3.addItem(spacerItem, 2, 3, 1, 1)
        self.comboBoxWeights = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.comboBoxWeights.setObjectName("comboBoxWeights")
        self.comboBoxWeights.addItem("")
        self.comboBoxWeights.addItem("")
        self.gridLayout_3.addWidget(self.comboBoxWeights, 5, 1, 1, 6)
        self.spinBoxStepK = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxStepK.setMinimum(1)
        self.spinBoxStepK.setMaximum(100)
        self.spinBoxStepK.setSingleStep(1)
        self.spinBoxStepK.setProperty("value", 1)
        self.spinBoxStepK.setObjectName("spinBoxStepK")
        self.gridLayout_3.addWidget(self.spinBoxStepK, 1, 6, 1, 1)
        self.checkBoxApplyClas = QtWidgets.QCheckBox(self.scrollAreaWidgetContents)
        self.checkBoxApplyClas.setObjectName("checkBoxApplyClas")
        self.gridLayout_3.addWidget(self.checkBoxApplyClas, 7, 0, 1, 2)
        self.labelStepK = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelStepK.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelStepK.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStepK.setObjectName("labelStepK")
        self.gridLayout_3.addWidget(self.labelStepK, 1, 5, 1, 1)
        self.spinBoxStartK = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxStartK.setMinimum(1)
        self.spinBoxStartK.setMaximum(100)
        self.spinBoxStartK.setProperty("value", 3)
        self.spinBoxStartK.setObjectName("spinBoxStartK")
        self.gridLayout_3.addWidget(self.spinBoxStartK, 1, 2, 1, 1)
        self.labelStartK = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelStartK.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelStartK.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStartK.setObjectName("labelStartK")
        self.gridLayout_3.addWidget(self.labelStartK, 1, 1, 1, 1)
        self.labelEndK = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelEndK.setAlignment(QtCore.Qt.AlignCenter)
        self.labelEndK.setObjectName("labelEndK")
        self.gridLayout_3.addWidget(self.labelEndK, 1, 3, 1, 1)
        self.spinBoxEndK = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.spinBoxEndK.setMinimum(1)
        self.spinBoxEndK.setMaximum(100)
        self.spinBoxEndK.setProperty("value", 12)
        self.spinBoxEndK.setObjectName("spinBoxEndK")
        self.gridLayout_3.addWidget(self.spinBoxEndK, 1, 4, 1, 1)
        self.labelkNN = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelkNN.setMaximumSize(QtCore.QSize(16777215, 20))
        self.labelkNN.setSizeIncrement(QtCore.QSize(0, 0))
        self.labelkNN.setBaseSize(QtCore.QSize(0, 0))
        self.labelkNN.setObjectName("labelkNN")
        self.gridLayout_3.addWidget(self.labelkNN, 1, 0, 1, 1)
        self.comboBoxMetric = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.comboBoxMetric.setObjectName("comboBoxMetric")
        self.comboBoxMetric.addItem("")
        self.comboBoxMetric.addItem("")
        self.comboBoxMetric.addItem("")
        self.gridLayout_3.addWidget(self.comboBoxMetric, 3, 1, 1, 6)
        self.labelMetric = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelMetric.setObjectName("labelMetric")
        self.gridLayout_3.addWidget(self.labelMetric, 3, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_3.addItem(spacerItem1, 6, 3, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_3.addItem(spacerItem2, 4, 3, 1, 1)
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
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem3, 0, 1, 1, 1)
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
        self.lineEditOutClass = QtWidgets.QLineEdit(self.layoutWidget1)
        self.lineEditOutClass.setObjectName("lineEditOutClass")
        self.gridLayout_2.addWidget(self.lineEditOutClass, 3, 0, 1, 1)
        self.buttonOutClass = QtWidgets.QPushButton(self.layoutWidget1)
        self.buttonOutClass.setObjectName("buttonOutClass")
        self.gridLayout_2.addWidget(self.buttonOutClass, 3, 1, 1, 1)
        self.tabWidget.addTab(self.tabOutput, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "k-NN classification"))
        self.buttonDataSetPath.setText(_translate("Dialog", "..."))
        self.labelTrain.setText(_translate("Dialog", "Training samples (Points)"))
        self.labelVal.setText(_translate("Dialog", "Validation samples (Points)"))
        self.labelClassTrain.setText(_translate("Dialog", "Class field - Training"))
        self.labelClassVal.setText(_translate("Dialog", " Class field - Validation "))
        self.labelDataSet.setText(_translate("Dialog", "Data set "))
        self.checkBoxPackPath_2.setText(_translate("Dialog", "Packages"))
        self.checkBoxPackPath.setText(_translate("Dialog", "Packages"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabInput), _translate("Dialog", "Input"))
        self.labelWeights.setText(_translate("Dialog", "Weights"))
        self.comboBoxWeights.setItemText(0, _translate("Dialog", "Uniform"))
        self.comboBoxWeights.setItemText(1, _translate("Dialog", "Distance"))
        self.checkBoxApplyClas.setText(_translate("Dialog", "Apply model"))
        self.labelStepK.setText(_translate("Dialog", "step"))
        self.labelStartK.setText(_translate("Dialog", "Start"))
        self.labelEndK.setText(_translate("Dialog", "end"))
        self.labelkNN.setText(_translate("Dialog", "k (n_neighbors)"))
        self.comboBoxMetric.setItemText(0, _translate("Dialog", "Manhattan (p=1)"))
        self.comboBoxMetric.setItemText(1, _translate("Dialog", "Euclidean (p=2)"))
        self.comboBoxMetric.setItemText(2, _translate("Dialog", "Minkowski (p=3)"))
        self.labelMetric.setText(_translate("Dialog", "Metrics (p)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabParameters), _translate("Dialog", "Parameters"))
        self.buttonAssessFile.setText(_translate("Dialog", "..."))
        self.buttonCancel.setText(_translate("Dialog", "Cancel"))
        self.buttonRun.setText(_translate("Dialog", "Run"))
        self.labelRFC.setText(_translate("Dialog", "Text file"))
        self.labelOutClass.setText(_translate("Dialog", "Vector file"))
        self.buttonOutClass.setText(_translate("Dialog", "..."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabOutput), _translate("Dialog", "Output"))

