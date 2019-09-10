# -*- coding: utf-8 -*-
"""
/***************************************************************************
 AccSeg
                                 A QGIS plugin
 Accuracy assessment measure of the segmentation
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2018-12-07
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
#To evaluate
from .to_evaluate import is_none, list_is_empty, exist_file, is_defined,is_crs, txt_is_writable

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .accuracy_segmentation_dialog import AccSegDialog
import os.path
import numpy as np

#Get layers QGIS
project_qgis3 = QgsProject.instance()

class AccSeg:
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
            'AccSeg_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = AccSegDialog()

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Accuracy of the segmentation')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'AccSeg')
        self.toolbar.setObjectName(u'AccSeg')

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
        return QCoreApplication.translate('AccSeg', message)


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

        icon_path = ':/plugins/accuracy_segmentation/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Accuracy of the segmentation'),
            callback=self.run,
            parent=self.iface.mainWindow())


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Accuracy of the segmentation'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar
        


    def run(self):
        """Run method that performs all the real work"""
        #Clear GUI
        self.dlg.ui.lineEditSegs.clear()
        self.dlg.ui.lineEditTextAss.clear()
        self.dlg.ui.comboBoxRefDat.clear()
        self.dlg.ui.labelOut.clear()
        #Zero  progress bar
        self.dlg.ui.progressBar.setValue(0)
        #Get layers QGIS
        self.dict_layers={'None':None}

        #Get name layers QGIS
        dic_layers_qgis =  project_qgis3.mapLayers()
        #Loop layers
        for name_layer in dic_layers_qgis.keys():
            #Append name and layer
            self.dict_layers[dic_layers_qgis[name_layer].name()]=dic_layers_qgis[name_layer]
            #print (name_layer)
            #assess if is raster
            if dic_layers_qgis[name_layer].type() == 1:
                 pass
             
            #assess if is vector    
            else:
                
                self.dlg.ui.comboBoxRefDat.addItem(dic_layers_qgis[name_layer].name())
         
        #Insert None comboBox if empty
        if self.dlg.ui.comboBoxRefDat.count()==0:
            self.dlg.ui.comboBoxRefDat.addItem('None')       
        #Connect functions
        self.dlg.ui.buttonSegs.clicked.connect(self.set_path_segs)
        self.dlg.ui.buttonTextAss.clicked.connect(self.set_path_text)
        self.dlg.ui.buttonCancel.clicked.connect(self.cancel_GUI)
        self.dlg.ui.buttonRun.clicked.connect(self.run_accsegs)         
        
        
        # show the dialog
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            
            pass
        
    def set_path_segs(self):
        #Clear
        self.dlg.ui.lineEditSegs.clear()
        #Open Directory
        pathSegs = QFileDialog.getExistingDirectory(None, self.tr('Open a folder'), None, QFileDialog.ShowDirsOnly)  
        #Set lineEditPathSegs
        self.dlg.ui.lineEditSegs.setText(pathSegs)
        return 0
    
    def set_path_text(self):
        #Clear
        self.dlg.ui.lineEditTextAss.clear()
        #Save assess file
        assess_file=QFileDialog.getSaveFileName(None, self.tr('Save file'), None, " Text file (*.txt)")
        
        #print (dir(assess_file[-1]))
        #get name and extension file
        name_file=assess_file[0].split(os.sep)[-1]
        #Assess extension
        if assess_file[0]=='':
            self.dlg.ui.lineEditTextAss.clear()
            
        elif assess_file[-1].split('*')[-1].endswith('.txt)'):
            #insert extension
            if name_file.endswith('.txt'):
                self.dlg.ui.lineEditTextAss.setText(assess_file[0])
            else:
                self.dlg.ui.lineEditTextAss.setText(assess_file[0]+'.txt')
        return 0
    
    def cancel_GUI(self):
        #disconnect        
        self.dlg.ui.buttonSegs.clicked.disconnect(self.set_path_segs)
        self.dlg.ui.buttonTextAss.clicked.disconnect(self.set_path_text)
        self.dlg.ui.buttonCancel.clicked.disconnect(self.cancel_GUI)
        self.dlg.ui.buttonRun.clicked.disconnect(self.run_accsegs)
        
        self.dlg.close()
        return 0
    
    def run_accsegs (self):
        self.dlg.ui.labelOut.clear()
        #progressbar                
        self.dlg.ui.progressBar.setValue(0)
        print('Run accuracy of the segmentation')
        #Assess lineEdit and comb
        if is_none(self.dlg.ui.comboBoxRefDat.currentText(),'Reference data is None'):
            return 0  
        
        #Assess lineEdit Segmentation
        elif is_defined(self.dlg.ui.lineEditSegs.text(),'Segmentation is not defined'):
            return 0
        
        #Assess lineEdit Segmentation
        elif is_defined(self.dlg.ui.lineEditTextAss.text(),'Text assessment is not defined'):
            return 0
        
        #Assess lineEdit and combobox data set
        elif exist_file(self.dlg.ui.lineEditSegs.text(),'Segmentation is not exist (Path error)'):
            return 0
        
        elif txt_is_writable(self.dlg.ui.lineEditTextAss.text(),'Error writing Text assessment'):
            return 0
        
        else:
            #Get variables
            #Get name and path reference data
            ref_data = self.dlg.ui.comboBoxRefDat.currentText()
            ref_data_path=self.dict_layers[ref_data].dataProvider().dataSourceUri().split('|')[0]
            #get names segmentations
            segs_path=self.dlg.ui.lineEditSegs.text()
            segs_names=[f for f in os.listdir(segs_path)  if f.endswith('.shp')]
            #Assessment empty list
            if list_is_empty(segs_names,'Segmentation folder empty: '+segs_path):
                return 0
            #Creat text assessment
            f_txt=self.dlg.ui.lineEditTextAss.text()
            txt_assess = open(f_txt,"w") 
            txt_assess.write('shape;'+'D'+';overSeg'+';underSeg'+'\n')
            #Set value progressbar

            totale=len(segs_names)
            count=1
            #Loop about segmentations
            for seg in segs_names :
                #Set value progressbar
                self.dlg.ui.progressBar.setValue((float(count)/float(totale))*100)
                count+=1
                #Read shapefile segmentation
                layer_seg = QgsVectorLayer(segs_path+os.sep+seg, seg, 'ogr')
                #print(segs_path,'----',seg)
                #Creat numpy array to store areas
                area_inters = np.array([])
                area_vals=np.array([])
                area_objs=np.array([])
                #Get features segmentacao
                features_seg = layer_seg.getFeatures()
                #Create spatial index
                index = QgsSpatialIndex()
                for feats_seg in features_seg:
                     index.insertFeature(feats_seg)
                #Get features reference data
                features_ref = self.dict_layers[ref_data].getFeatures()
                #Loop reference data
                for feat_val in features_ref:        
                    #Obter os ids pela interseccao entre obj e val
                    ids=index.intersects(feat_val.geometry().centroid().boundingBox())
                    #create list vazia
                    area_intersect=[]
                    area_obj=[]
                    #Selecionar o objs com ids
                    layer_seg.selectByIds(ids)
                    #print (ids)
                    #features selecionadas
                    features = layer_seg.selectedFeatures()
                    #percorrer features selecionadas
                    for obj in features:
                        area_intersect.append(feat_val.geometry().intersection(obj.geometry()).area())
                        area_obj.append(obj.geometry().area())
                    #Get areas, intersection, reference data and objects   
                    area_inters = np.append(area_inters,max(area_intersect))
                    area_vals=np.append(area_vals,area_obj[area_intersect.index(max(area_intersect))])
                    area_objs=np.append(area_objs,feat_val.geometry().area())
            
                #Calcular qualidade segmentacao
                overSegmentation = 1.- (sum(area_inters)/sum(area_vals))
                underSegmentation = 1. - (sum(area_inters)/sum(area_objs))
                d=np.sqrt(overSegmentation**2+underSegmentation**2)
                txt_assess.write(seg+';'+str(round(d,3))+';'+str(round(overSegmentation,3))+';'+str(round(underSegmentation,3))+'\n')
                print(seg +' - D: '+str(round(d,3)))
                self.dlg.ui.labelOut.setText(seg +' - D: '+str(round(d,3)))
                
        #set value progress ba
        self.dlg.ui.progressBar.setValue(100)
        print('Finalizou')  
        self.dlg.ui.labelOut.setText('Finish')
        txt_assess.close()                