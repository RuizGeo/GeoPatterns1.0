# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Naive Bayes
                                 A QGIS plugin
Naive Bayes classificaton
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2018-05-14
        git sha              : $Format:%H$
        copyright            : (C) 2018 by Luis Fernando Chimelo Ruiz
        email                : ruiz.ch@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qgis.core import *
import numpy as np
import os
import sys
#import time

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .NB_dialog import NBDialog
import os.path
from .NBC import model_NB
from .to_evaluate import is_none, is_defined, exist_file,\
list_is_empty, txt_is_writable,field_is_integer,field_is_real,\
vector_is_readable,is_crs,is_join

#Get layers QGIS
project = QgsProject.instance()

class NB:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'kNN_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = NBDialog()
        #Append packages path in sys
        if self.dlg.ui.lineEditPackPath.text()=='':
            pass
        else:
            
            sys.path.append(self.dlg.ui.lineEditPackPath.text())
        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&GeoPatterns')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'Naive Bayes')
        self.toolbar.setObjectName(u'Naive Bayes')

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('Naive Bayes', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/naive_bayes/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'k-NN'),
            callback=self.run,
            parent=self.iface.mainWindow())


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&k-NN'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar


    def run(self):
        """Run method that performs all the real work"""
        #Get plugin path
        self.plugin_path= QgsApplication.qgisSettingsDirPath()+'python/plugins/naive_bayes'
        #Try disconnect
        try:
            #disconnect       
            self.dlg.ui.buttonAssessFile.clicked.disconnect(self.set_assess_file)
            self.dlg.ui.buttonDataSetPath.clicked.disconnect(self.set_dataset_path)
            self.dlg.ui.buttonCancel.clicked.disconnect(self.cancel_GUI)
            self.dlg.ui.buttonRun.clicked.disconnect(self.run_classification)
            self.dlg.ui.comboBoxTrain.currentIndexChanged.disconnect(self.value_changed_train)
            self.dlg.ui.comboBoxVal.currentIndexChanged.disconnect(self.value_changed_val)
            self.dlg.ui.checkBoxApplyClas.stateChanged.disconnect(self.state_changed_apply_class)
            self.dlg.ui.buttonOutClass.clicked.disconnect(self.set_classification_path)
            self.dlg.ui.checkBoxPackPath.stateChanged.disconnect(self.state_changed_packages_path)
        except:
            pass
        #TabInput visble
        self.dlg.ui.tabWidget.setCurrentIndex(0)
        #Clear GUIs
        self.dlg.ui.comboBoxClassTrain.clear()
        self.dlg.ui.comboBoxClassVal.clear()
        self.dlg.ui.comboBoxPriori.clear()
        self.dlg.ui.comboBoxVal.clear()
        self.dlg.ui.lineEditDataSetPath.clear()
        self.dlg.ui.lineEditOutClass.clear()
        self.dlg.ui.lineEditAssessFile.clear()
        #SetCheckState
        self.dlg.ui.checkBoxApplyClas.setCheckState(False)

        #enable
        self.dlg.ui.lineEditOutClass.setEnabled(False)
        self.dlg.ui.buttonOutClass.setEnabled(False)
        #Zero progressaBar
        self.dlg.ui.progressBar.setValue(0)
        #Set criterions Split comboBoxPriori
        self.dlg.ui.comboBoxPriori.addItems(['True','False'])

        #Creat Dict with layers names name:layer
        self.dict_layers={'None':None}
        #Insert fields comboBoxs
        self.fields={}
        #Dict with layers names
        dic_layers_qgis =  project.mapLayers()
        for name_layer in dic_layers_qgis.keys():
            #Append name and layer
            self.dict_layers[dic_layers_qgis[name_layer].name()]=dic_layers_qgis[name_layer]
            print (name_layer)
            #assess if is vector
            if dic_layers_qgis[name_layer].type() == 0:
                 #Get field names
                 field_names = [field.name() for field in dic_layers_qgis[name_layer].dataProvider().fields() ]
                 self.fields[dic_layers_qgis[name_layer].name()]=field_names
                 #add name vectors
                 self.dlg.ui.comboBoxTrain.addItem(dic_layers_qgis[name_layer].name())    
                 self.dlg.ui.comboBoxVal.addItem(dic_layers_qgis[name_layer].name())
            else:
                pass
        #Insert values ComboBoxs                
        if self.dlg.ui.comboBoxTrain.count() == 0:   
            self.dlg.ui.comboBoxTrain.addItem('None')  
            self.dlg.ui.comboBoxClassTrain.addItem('None') 
        else:
            #Get name layer
            names_fields=self.dlg.ui.comboBoxTrain.currentText()           
            #Get and field layer
            self.dlg.ui.comboBoxClassTrain.addItems(self.fields[names_fields])
            
            
        if self.dlg.ui.comboBoxVal.count() == 0:             
            self.dlg.ui.comboBoxVal.addItem('None') 
            self.dlg.ui.comboBoxClassVal.addItem('None')
        else:
            #Get name layer
            names_fields=self.dlg.ui.comboBoxVal.currentText() 
            self.dlg.ui.comboBoxClassVal.addItems(self.fields[names_fields])
        #print packages
        print(self.plugin_path+os.sep+'packages_path.txt')
        #Append packages path in sys
        if self.dlg.ui.lineEditPackPath.text()=='':
            pass
        else:
            
            sys.path.append(self.dlg.ui.lineEditPackPath.text())
        #Read file packages_path.txt
        txt_packages_path=open(self.plugin_path+os.sep+'packages_path.txt','r')
        #Replace new line ('\n') for ''
        packages_path= [line.replace('\n', '') for line in txt_packages_path.readlines()]
        
        print(packages_path)
        #Set packages path in lineEdit
        if packages_path == []:
            pass
        else:
            self.dlg.ui.lineEditPackPath.setText(packages_path[0])
        #Close packages_path.txt
        txt_packages_path.close()             
        #Connect functions
        self.dlg.ui.buttonDataSetPath.clicked.connect(self.set_dataset_path)
        self.dlg.ui.buttonAssessFile.clicked.connect(self.set_assess_file)
        self.dlg.ui.buttonCancel.clicked.connect(self.cancel_GUI)
        self.dlg.ui.buttonRun.clicked.connect(self.run_classification)
        self.dlg.ui.buttonOutClass.clicked.connect(self.set_classification_path)

        #Connect functions valueChanged comboBox
        self.dlg.ui.comboBoxTrain.currentIndexChanged.connect(self.value_changed_train)
        self.dlg.ui.comboBoxVal.currentIndexChanged.connect(self.value_changed_val)
        #Connect functions CheckBox
        self.dlg.ui.checkBoxApplyClas.stateChanged.connect(self.state_changed_apply_class)
        self.dlg.ui.checkBoxPackPath.stateChanged.connect(self.state_changed_packages_path)
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass
        
    def state_changed_packages_path(self):

        state=self.dlg.ui.checkBoxPackPath.checkState()     
        print ('State packages path: ',state)
        if state ==2:

            #Enable line edit
            self.dlg.ui.lineEditPackPath.setEnabled(True)

            
        else:            
            if self.dlg.ui.lineEditPackPath.text()=='':
                #Read file packages_path.txt
                txt_packages_path=open(self.plugin_path+os.sep+'packages_path.txt','w')
                txt_packages_path.write('')
                txt_packages_path.close()
                #self.dlg.ui.lineEditPackPath.setText('None')
                self.dlg.ui.lineEditPackPath.setEnabled(False)
                
            else:

                #Assess lineEdit and combobox data set
                if exist_file(self.dlg.ui.lineEditPackPath.text(),'Input: packages path is not exist'):
                    self.dlg.ui.lineEditPackPath.clear()
                    #Write file packages_path.txt
                    txt_packages_path=open(self.plugin_path+os.sep+'packages_path.txt','w')
                    txt_packages_path.write('')
                    txt_packages_path.close()
                    self.dlg.ui.checkBoxPackPath.setChecked(False) 
                    self.dlg.ui.lineEditPackPath.setEnabled(False)
                    return 0
                else:
                    #Write file packages_path.txt
                    txt_packages_path=open(self.plugin_path+os.sep+'packages_path.txt','w')
                    txt_packages_path.write(self.dlg.ui.lineEditPackPath.text())
                    txt_packages_path.close()
                    self.dlg.ui.lineEditPackPath.setEnabled(False)
                    #Set packages path in sys
                    sys.path.append(self.dlg.ui.lineEditPackPath.text())   
                    
    def state_changed_apply_class (self):
        state=self.dlg.ui.checkBoxApplyClas.checkState()     
        print ('State apply classification: ',state)
        if state ==2:
            self.dlg.ui.lineEditOutClass.setEnabled(True)
            self.dlg.ui.buttonOutClass.setEnabled(True)
        else: 
            self.dlg.ui.lineEditOutClass.setEnabled(False)
            self.dlg.ui.buttonOutClass.setEnabled(False)    
            
    def value_changed_train(self):
        
        #Clear
        self.dlg.ui.comboBoxClassTrain.clear()      
        #Get name layer
        name_layer=self.dlg.ui.comboBoxTrain.currentText() 
        #Get field names
        #field_names = [field.name() for field in self.dict_layers[name_layer].dataProvider().fields() ]
        #Get and field layer
        self.dlg.ui.comboBoxClassTrain.addItems(self.fields[name_layer])    
        print (self.fields[name_layer])
        return 0
    
    def value_changed_val (self):
        #Clear
        self.dlg.ui.comboBoxClassVal.clear()      
        #Get name layer
        name_layer=self.dlg.ui.comboBoxVal.currentText() 
        #Get field names
        #field_names = [field.name() for field in self.dict_layers[name_layer].dataProvider().fields() ]
        #Get and field layer
        self.dlg.ui.comboBoxClassVal.addItems(self.fields[name_layer])    
        print (self.fields[name_layer])
        return 0 
    def set_dataset_path (self):
        #Open Directory
        self.dataset_path = QFileDialog.getExistingDirectory(None, self.tr('Open a folder'), None, QFileDialog.ShowDirsOnly)  
        #Set lineEditPathSegs
        self.dlg.ui.lineEditDataSetPath.setText(self.dataset_path)
        return 0
    def set_assess_file(self):
        #Clear
        self.dlg.ui.lineEditAssessFile.clear()
        #Save assess file
        assess_file=QFileDialog.getSaveFileName(None, self.tr('Save file'), None, " Text file (*.txt);;Comma-separated values (*.csv)")
        print(assess_file)
        #print (dir(assess_file[-1]))
        #get name and extension file
        name_file=assess_file[0].split(os.sep)[-1]
        #Assess extension
        if assess_file[0]=='':
            self.dlg.ui.lineEditAssessFile.clear()
            
        elif assess_file[-1].split('*')[-1].endswith('.txt)'):

            #insert extension
            if name_file.endswith('.txt'):
                self.dlg.ui.lineEditAssessFile.setText(assess_file[0])
            else:
                self.dlg.ui.lineEditAssessFile.setText(assess_file[0]+'.txt')
        else:
            if name_file.endswith('.csv'):
                self.dlg.ui.lineEditAssessFile.setText(assess_file[0])
            else:
                self.dlg.ui.lineEditAssessFile.setText(assess_file[0]+'.csv')
        
    def set_classification_path (self):
        #clear lineEdit
        self.dlg.ui.lineEditOutClass.clear()
        #Save shape classification
        out_file = QFileDialog.getSaveFileName(None, self.tr('Save file'), None, " Shapefile (*.shp)")
        #get name and extension file
        name_file= out_file[0].split(os.sep)[-1]
        #Assess extension
        if out_file[-1].split('*')[-1].endswith('.shp)'):
            #insert extension
            if name_file.endswith('.shp'):
                self.dlg.ui.lineEditOutClass.setText(out_file[0])
            else:
                self.dlg.ui.lineEditOutClass.setText(out_file[0]+'.shp')
        else:
            pass

    
    def cancel_GUI(self):
        
        #disconnect       
        self.dlg.ui.buttonAssessFile.clicked.disconnect(self.set_assess_file)
        self.dlg.ui.buttonDataSetPath.clicked.disconnect(self.set_dataset_path)
        self.dlg.ui.buttonCancel.clicked.disconnect(self.cancel_GUI)
        self.dlg.ui.buttonRun.clicked.disconnect(self.run_classification)
        self.dlg.ui.comboBoxTrain.currentIndexChanged.disconnect(self.value_changed_train)
        self.dlg.ui.comboBoxVal.currentIndexChanged.disconnect(self.value_changed_val)
        self.dlg.ui.checkBoxApplyClas.stateChanged.disconnect(self.state_changed_apply_class)
        self.dlg.ui.buttonOutClass.clicked.disconnect(self.set_classification_path)
        self.dlg.ui.checkBoxPackPath.stateChanged.disconnect(self.state_changed_packages_path)

        self.dlg.close()
        
        return 0
        
    def run_classification(self):
        print('Run classification')
        #Assess lineEdit and combobox
        if self.dlg.ui.comboBoxTrain.currentText() == 'None':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Input: Select training samples")
            msg.setWindowTitle("Info")
            msg.exec_() 
            return 0 
        
        elif self.dlg.ui.comboBoxVal.currentText() == 'None':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Input: Select validation samples")
            msg.setWindowTitle("Info")
            msg.exec_() 
            return 0 
        
        elif self.dlg.ui.lineEditDataSetPath.text() == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Input: Undefined segmentation path")
            msg.setWindowTitle("Info")
            msg.exec_() 
            return 0 
        
        elif self.dlg.ui.lineEditOutClass.text() == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Output: Undefined classification path")
            msg.setWindowTitle("Info")
            msg.exec_() 
            return 0 
        
        elif not os.path.exists(self.dlg.ui.lineEditDataSetPath.text()):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Input: The path does not exist")
            msg.setWindowTitle("Info")
            msg.exec_() 
            return 0 
        else:
            #Classification path
            classification_path=self.dlg.ui.lineEditOutClass.text()
            #Get name and path layer - Input trainining samples
            name_train = self.dlg.ui.comboBoxTrain.currentText()
            train_path=self.dict_layers[name_train].dataProvider().dataSourceUri().split('|')[0]
            #Get name field class - training samples
            field_class_train = self.dlg.ui.comboBoxClassTrain.currentText()
            #Type name field (train)
            fields_train=self.dict_layers[name_train].fields()
            type_field_train = fields_train[fields_train.indexFromName(field_class_train)].typeName()
            print('Type field class (training): '+type_field_train)
            #Get name and path layer - Input validation samples
            name_val=self.dlg.ui.comboBoxVal.currentText()
            validation_path=self.dict_layers[name_val].dataProvider().dataSourceUri().split('|')[0]
            #Get name field class - training samples
            field_class_val = self.dlg.ui.comboBoxClassVal.currentText()
            #Type name field (Val)
            fields_val=self.dict_layers[name_val].fields()
            type_field_val = fields_val[fields_val.indexFromName(field_class_val)].typeName()
            print('Type field class (validation): '+type_field_val)

            #data set path
            dataset_path=self.dlg.ui.lineEditDataSetPath.text()             

            #Get Metric
            a_priori = self.dlg.ui.comboBoxPriori.currentText()


            #Assess exist files
            if not os.path.exists(train_path):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Input: The file does not exist: "+train_path)
                msg.setWindowTitle("Info")
                msg.exec_() 
                return 0 
            if not os.path.exists(validation_path):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Input: The file does not exist: "+validation_path)
                msg.setWindowTitle("Info")
                msg.exec_() 
                return 0 
            #Get names files segmentations
            dataset_names=[f for f in os.listdir(dataset_path)  if f.endswith('.shp')]
            if  dataset_names==[]:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Input: Path Doesn't contain shapefile: "+dataset_path)
                msg.setWindowTitle("Info")
                msg.exec_() 
                return 0   
            #Get path assess file
            assess_text_path=self.dlg.ui.lineEditAssessFile.text()
            if  assess_text_path=='':
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Output: Undefined Text path: "+assess_text_path)
                msg.setWindowTitle("Info")
                msg.exec_() 
                return 0   
               
            #CheckBox Apply classification
            if self.dlg.ui.checkBoxApplyClas.checkState()==2:
                state_applyclass= 'True'
                #Assess classification path
                if  classification_path=='':
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText("Output: Undefined classification path: "+classification_path)
                    msg.setWindowTitle("Info")
                    msg.exec_() 
                    return 0 
            else:
                state_applyclass= 'False'
            #arg[1]=train_path

            #progressbar
            self.dlg.ui.progressBar.setValue(50)



            #Run K-NN
            model_NB(self.dlg.ui.progressBar,train_path,dataset_path,\
                        validation_path,field_class_train,\
                        field_class_val,a_priori,state_applyclass,\
                        assess_text_path,classification_path)


            print('Finish')
            #progressbar                
            self.dlg.ui.progressBar.setValue(100)
            #self.dlg.ui.progressBar.setTextVisible(True)
            #self.dlg.ui.progressBar.setFormat('Finish')
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('Finish')
            msg.setWindowTitle("Info")
            msg.exec_()      