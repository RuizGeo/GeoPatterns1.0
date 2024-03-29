# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Segmentation
                                 A QGIS plugin
 Segmentation GRASS GIS
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2018-02-23
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
import processing
import time
#from qgis.PyQt.QtWidgets import QFileDialog


#Get layers QGIS
project = QgsProject.instance()



# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .segmentation_dialog import SegmentationDialog
import os.path
from .to_evaluate import is_arange

class Segmentation:
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
            'Segmentation_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = SegmentationDialog()

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&GeoPatterns')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'Segmentation')
        self.toolbar.setObjectName(u'Segmentation')
        
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
        return QCoreApplication.translate('Segmentation', message)


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
    
            icon_path = ':/plugins/segmentation/icon.png'
            self.add_action(
                icon_path,
                text=self.tr(u'Segmentation'),
                callback=self.run,
                parent=self.iface.mainWindow())
            

        


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginRasterMenu(
                self.tr(u'&Segmentation'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar
        
        #self.dlg.ui.buttonPathSegs.clicked.disconnect(self.set_path_segs)
        #self.dlg.ui.buttonCancel.clicked.disconnect(self.cancel_GUI)
        #self.dlg.ui.buttonRun.clicked.disconnect(self.run_segmentation)
        print ('unload')
        return 0


    def run(self):
        
        """Run method that performs all the real work"""

        #try disconnect
        try:
            #Connect functions
            self.dlg.ui.buttonPathSegs.clicked.disconnect(self.set_path_segs)
            self.dlg.ui.buttonCancel.clicked.disconnect(self.cancel_GUI)
            self.dlg.ui.buttonRun.clicked.connect(self.run_segmentation)
            
            #Connect functions value changedspinBox
            self.dlg.ui.spinBoxStarSim.valueChanged.disconnect(self.value_changed_start_sim)
            self.dlg.ui.spinBoxEndSim.valueChanged.disconnect(self.value_changed_end_sim)
            self.dlg.ui.spinBoxStepSim.valueChanged.disconnect(self.value_changed_step_sim)
            #self.dlg.ui.spinBoxStepSim.valueChanged.connect(self.value_changed_step_sim)
            self.dlg.ui.spinBoxStartMcells.valueChanged.disconnect(self.value_changed_start_Mcells)
            self.dlg.ui.spinBoxEndMcells.valueChanged.disconnect(self.value_changed_end_Mcells)
            self.dlg.ui.spinBoxStepMcells.valueChanged.disconnect(self.value_changed_step_Mcells)
            self.dlg.ui.comboBoxRaster.currentIndexChanged.disconnect() 
            
        except:
            pass
        # Do something useful here - delete the line containing pass and
        # substitute with your code.
        #TabInput visble
        self.dlg.ui.tabWidget.setCurrentIndex(0)
        #Clean GUIs
        self.dlg.ui.comboBoxRaster.clear()
        self.dlg.ui.comboBoxNeigh.clear()
        #self.dlg.ui.comboBoxSeeds.clear()
        self.dlg.ui.comboBoxSimMeth.clear()
        #TextEdit
        self.dlg.ui.textEditOut.clear()
        #self.dlg.ui.textEditInput.clear()

        self.dlg.ui.lineEditPathSegs.clear()
        
        #Zero  progress bar
        self.dlg.ui.progressBar.setValue(0)
        

        self.dlg.ui.comboBoxSimMeth.addItems(['euclidean', 'manhattan'])
        self.dlg.ui.comboBoxNeigh.addItems(['4','8'])

        #Add values GUIs
        self.dict_layers={'None':None}
        ##Seeds and bounding##
        #self.dlg.ui.comboBoxSeeds.addItem('None')
        #self.dlg.ui.comboBoxBounding.addItem('None')
        #Get name layers QGIS
        dic_layers_qgis =  project.mapLayers()
        for name_layer in dic_layers_qgis.keys():
            #Append name and layer
            self.dict_layers[dic_layers_qgis[name_layer].name()]=dic_layers_qgis[name_layer]
            print (name_layer)
            #assess if is raster
            if dic_layers_qgis[name_layer].type() == 1:
                 #add name rasters
                 self.dlg.ui.comboBoxRaster.addItem(dic_layers_qgis[name_layer].name())
                 #self.dlg.ui.comboBoxSeeds.addItem(dic_layers_qgis[name_layer].name())
                 #self.dlg.ui.comboBoxBounding.addItem(dic_layers_qgis[name_layer].name())
             #assess if is vector    
   
            else:
                pass
   
        #Insert None Value Combo Raster and SHP validation
           
        if self.dlg.ui.comboBoxRaster.count() == 0:   
            self.dlg.ui.comboBoxRaster.addItem('None')      
        
        #Connect functions
        self.dlg.ui.buttonPathSegs.clicked.connect(self.set_path_segs)
        self.dlg.ui.buttonCancel.clicked.connect(self.cancel_GUI)
        self.dlg.ui.buttonRun.clicked.connect(self.run_segmentation)
        
        #Connect functions value changedspinBox
        self.dlg.ui.spinBoxStarSim.valueChanged.connect(self.value_changed_start_sim)
        self.dlg.ui.spinBoxEndSim.valueChanged.connect(self.value_changed_end_sim)
        self.dlg.ui.spinBoxStepSim.valueChanged.connect(self.value_changed_step_sim)
        #self.dlg.ui.spinBoxStepSim.valueChanged.connect(self.value_changed_step_sim)
        self.dlg.ui.spinBoxStartMcells.valueChanged.connect(self.value_changed_start_Mcells)
        self.dlg.ui.spinBoxEndMcells.valueChanged.connect(self.value_changed_end_Mcells)
        self.dlg.ui.spinBoxStepMcells.valueChanged.connect(self.value_changed_step_Mcells)
        #
        
        #Insert assess import i.segment
        #parameters_iSeg=processing.algorithmHelp('grass7:i.segment')

           
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            pass
    def value_changed_step_sim(self):
        if self.dlg.ui.spinBoxStepSim.value() > self.dlg.ui.spinBoxEndSim.value():
            self.dlg.ui.spinBoxStepSim.setMaximum(self.dlg.ui.spinBoxEndSim.value())
            self.dlg.ui.spinBoxStepSim.setValue(self.dlg.ui.spinBoxEndSim.value())
        
    def value_changed_end_sim(self):
        self.dlg.ui.spinBoxStepSim.setMaximum(self.dlg.ui.spinBoxEndSim.value())
            
    def value_changed_start_sim (self):
        #Set minimum valspinBoxEndSim
        self.dlg.ui.spinBoxEndSim.setMinimum(self.dlg.ui.spinBoxStarSim.value())
    def value_changed_start_Mcells (self):
        #Set minimum valspinBoxEndSim
        #self.dlg.ui.spinBoxStartMcells.setMinimum(self.dlg.ui.spinBoxStepMcells.value())
       
        self.dlg.ui.spinBoxEndMcells.setMinimum(self.dlg.ui.spinBoxStartMcells.value())

    def value_changed_end_Mcells(self):
        self.dlg.ui.spinBoxStepMcells.setMaximum(self.dlg.ui.spinBoxEndMcells.value())
        
    def value_changed_step_Mcells(self):
        if self.dlg.ui.spinBoxStepMcells.value() > self.dlg.ui.spinBoxEndMcells.value():
            self.dlg.ui.spinBoxStepMcells.setMaximum(self.dlg.ui.spinBoxEndMcells.value())
    def set_path_segs(self):
        #Open Directory
        self.pathSegs = QFileDialog.getExistingDirectory(None, self.tr('Open a folder'), None, QFileDialog.ShowDirsOnly)  
        #Set lineEditPathSegs
        self.dlg.ui.lineEditPathSegs.setText(self.pathSegs)
        return 0
        

    def cancel_GUI(self):
        
        
        
        #disconnect
        
        #Connect functions
        self.dlg.ui.buttonPathSegs.clicked.disconnect(self.set_path_segs)
        self.dlg.ui.buttonCancel.clicked.disconnect(self.cancel_GUI)
        self.dlg.ui.buttonRun.clicked.connect(self.run_segmentation)
        
        #Connect functions value changedspinBox
        self.dlg.ui.spinBoxStarSim.valueChanged.disconnect(self.value_changed_start_sim)
        self.dlg.ui.spinBoxEndSim.valueChanged.disconnect(self.value_changed_end_sim)
        self.dlg.ui.spinBoxStepSim.valueChanged.disconnect(self.value_changed_step_sim)
        #self.dlg.ui.spinBoxStepSim.valueChanged.connect(self.value_changed_step_sim)
        self.dlg.ui.spinBoxStartMcells.valueChanged.disconnect(self.value_changed_start_Mcells)
        self.dlg.ui.spinBoxEndMcells.valueChanged.disconnect(self.value_changed_end_Mcells)
        self.dlg.ui.spinBoxStepMcells.valueChanged.disconnect(self.value_changed_step_Mcells)
        #self.dlg.ui.comboBoxRaster.currentIndexChanged.disconnect(self.value_changed_raster)
        #del
        self.dlg.close()
        return 0
        
    def run_segmentation(self):
        #Get layer comboBoxRaster - Input rasters
        if self.dlg.ui.comboBoxRaster.currentText() == 'None':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Input: Undefined raster input (None)")
            msg.setWindowTitle("Info")
            msg.exec_() 
            return 0 
        else:
            name_input_raster = self.dlg.ui.comboBoxRaster.currentText()
            #Get extent raster
            ext = self.dict_layers[name_input_raster].extent()
            xmax,xmin, ymax , ymin= ext.xMaximum(), ext.xMinimum(), ext.yMaximum(), ext.yMinimum()
            self.dlg.ui.textEditOut.setText('xmax, xmin, ymax , ymin: '+', '+str(xmax)+ ', '+str(xmin)+', '+ str(ymax) +', '+str(ymin))
            #Get size cells
            size_cell= self.dict_layers[name_input_raster].rasterUnitsPerPixelX()
            self.dlg.ui.textEditOut.append('cell size: '+str(size_cell))
            #Get path input raster 
            path_input_raster=self.dict_layers[name_input_raster].dataProvider().dataSourceUri()
            print ('Run: ',self.dict_layers[name_input_raster].name())
          
            #Assess files segs and text
            paths=self.dlg.ui.lineEditPathSegs.text()
            
            if paths == '' or \
            not os.path.exists(paths):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Output: Undefined path")
                msg.setWindowTitle("Info")
                msg.exec_() 
                return 0
            else:
                #Create directory Segs, Goods and SHP
                path_segs=paths+os.path.sep+'IMGs'                 
                #Goods
                path_goods= paths+os.path.sep+'goods'                   
                #SHP
                path_shps=paths+os.path.sep+'shapes'
                
                #Asses files exist
                if os.path.exists(path_segs) or os.path.exists(path_goods) or\
                os.path.exists(path_shps):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText("Output: A folder already exists (shapes, goods or IMGs) ")
                    msg.setWindowTitle("Info")
                    msg.exec_() 
                    return 0  
                  
                else:
                    #Create directory Segs, Goods and SHP
                    os.makedirs(path_segs)
                    os.makedirs(path_goods)
                    os.makedirs(path_shps)
                    
                    #Get layer comboBoxSeeds - Input seeds
                    #input_seeds= self.dlg.ui.comboBoxSeeds.currentText()
                    #Get layer comboBoxBounding - Input Bounding
                    #input_bounding= self.dlg.ui.comboBoxBounding.currentText()  
                    
                    #Get method 
                    method_similarity={'euclidean':0, 'manhattan':1}
                    meth_sim = method_similarity[self.dlg.ui.comboBoxSimMeth.currentText()]
                    neighbors=int(self.dlg.ui.comboBoxNeigh.currentText())
                    print(self.dlg.ui.comboBoxSimMeth.currentText(),neighbors)

                    use_memory=self.dlg.ui.spinBoxMemory.value()
                    #Get iterations
                    iteratios=self.dlg.ui.spinBoxMiter.value()
                    #Get valures Similarity
                    start_sim=self.dlg.ui.spinBoxStarSim.value()
                    end_sim=self.dlg.ui.spinBoxEndSim.value()
                    step_sim=self.dlg.ui.spinBoxStepSim.value()
                    
                    #Get valures minimun cells segments
                    end_mcells=self.dlg.ui.spinBoxEndMcells.value()
                    start_mcells=self.dlg.ui.spinBoxStartMcells.value()
                    step_mcells=self.dlg.ui.spinBoxStepMcells.value()
  
                    #Where star equal 0, init equal step
                    end_sim=end_sim+step_sim
                    if start_sim==0.0:
                        start_sim=step_sim
                        
                    if end_sim > 1:
                        end_sim=1
                    if start_mcells==0.0:
                        start_mcells=step_mcells
                    
                    #Size arange parameters
                    size_sim=np.arange(start_sim,end_sim,step_sim).size
                    size_cells=np.arange(start_mcells,end_mcells+step_mcells,step_mcells).size
                    size_param=size_sim*size_cells
                    
                    ###########  i.segment  ##############################
                    
                
                    #somar loop
                    count=1
                    #set value progressBar
                    self.dlg.ui.progressBar.setValue(1)
                    #size != []
                    similarity_calculatiom=np.arange(start_sim,end_sim,step_sim)
                 
                    if  is_arange(similarity_calculatiom, 'Parameters: Similarity threshold is empty:'+str(similarity_calculatiom)+' '):
                        return 0    
                    for sim in np.arange(start_sim,end_sim,step_sim):
                    
                        #4 decimals 
                        sim = round(sim,4)
                        
                        for mcell in np.arange(start_mcells,end_mcells+step_mcells,step_mcells):
    
                            print (sim,mcell)
                            #Path output segmentation
                            path_out_seg = path_segs+os.path.sep+'seg_'+str(sim).replace('.','')+'_'+str(mcell)+'.tif'
                            #path output segmentation vector
                            path_out_seg_vec= path_shps+os.path.sep+'seg_'+str(sim).replace('.','')+'_'+str(mcell)+'.shp'
                            #path output segmentations goods
                            path_out_goods = path_goods+os.path.sep+'seg_'+str(sim).replace('.','')+'_'+str(mcell)+'.tif'
                            
                            try:
                                #init time
                                ini = time.time()
                                #set parameters i.segment
                                parameters = {"input":path_input_raster,"threshold":float(sim),\
                                              "method":0,"similarity":float(meth_sim),"minsize":float(mcell),"memory":float(use_memory),\
                                              "iterations":iteratios,"seeds":None,"bounds":None,\
                                              "GRASS_REGION_PARAMETER":"%f,%f,%f,%f" % (xmin, xmax, ymin, ymax),\
                                              "GRASS_REGION_CELLSIZE_PARAMETER":size_cell,\
                                              "output":path_out_seg,\
                                              "goodness":path_out_goods,\
                                              '-d':neighbors,'-w':False}
                                print (parameters)
                                #Run i.segment
                                processing.run('grass7:i.segment',parameters)
                                #Append textEdiOut
                                end=time.time()
                                self.dlg.ui.textEditOut.append('Threshold: '+str(sim)+ ' - '+'Size: '+str(mcell)+' - '+'Time: '+str(round((end-ini)/60.,4))+ ' min')
                                
                            except:
                                msg = QMessageBox()
                                msg.setIcon(QMessageBox.Critical)
                                msg.setText("Error: i.segment()")
                                msg.setWindowTitle("Error")
                                msg.exec_()
                                return 0
                            try:
                                #Parameters output seg SHP
                                parameters_shp={"INPUT":path_out_seg,"BAND":1,"FIELD":'id_seg',\
                                "EIGHT_CONNECTEDNESS":False,\
                                "OUTPUT":path_out_seg_vec}
                                #Polygonize
                                processing.run('gdal:polygonize', parameters_shp)
                               
                                #set value progressBar
                                self.dlg.ui.progressBar.setValue(int(float(count)/size_param)*100))
                                count +=1
                            except:
                                msg = QMessageBox()
                                msg.setIcon(QMessageBox.Critical)
                                msg.setText("Output: gdal:polygonize")
                                msg.setWindowTitle("Error")
                                msg.exec_() 
                                return 0
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText('Finish')
                    msg.setWindowTitle("Info")
                    msg.exec_()            
                #identado com o primeiro FOR         
                return 1
                    

                       
