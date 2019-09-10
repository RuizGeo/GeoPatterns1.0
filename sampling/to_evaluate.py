#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 18:55:25 2018

@author: ruiz
"""

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qgis.core import *

import os



import geopandas as gpd


#To evaluate lineEdit is None
def is_none(text,messenger):
    
        if text == 'None':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(messenger+" is None!!")
            msg.setWindowTitle("Info")
            msg.exec_() 
            return True 
        else:
            return False
        
#To evaluate lineEdit empty        
def is_defined(text,messenger):
    
        if text == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(messenger)
            msg.setWindowTitle("Info")
            msg.exec_() 
            return True
        else:
            return False
        
#To evaluate lineEdit exist file or folder   
def exist_file(text,messenger):   
     
        if not os.path.exists(text):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(messenger+': '+text)
            msg.setWindowTitle("Info")
            msg.exec_() 
            return True
        else:
            return False

#To evaluate list is empty       
def list_is_empty(lista,messenger):
    
            if  lista==[]:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(messenger)
                msg.setWindowTitle("Info")
                msg.exec_() 
                return True
            else:
                return False

#To evaluate text write      
def txt_is_writable(path,messenger):
            try:
                open(path,'w')
                
            except:               
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(messenger+': '+path)
                msg.setWindowTitle("Info")
                msg.exec_() 
                return True
            
            else:
                return False

#To evaluate field is integer     
def field_is_real(txt,messenger):
            if txt =='Real':
                
               return True
            
            else:
                             
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(messenger+' - '+txt)
                msg.setWindowTitle("Info")
                msg.exec_() 
                
                return False            
#To evaluate field is integer     
def field_is_integer(txt,messenger):
            if txt in ['Integer64','Integer']:
                
               return True
            
            else:
                             
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(messenger+' - '+txt)
                msg.setWindowTitle("Info")
                msg.exec_() 
                
                return False
        
#To evaluate vector readable     
def vector_is_readable(path,messenger):
    
    layer = QgsVectorLayer(path)
    
    if layer.isValid():
        return True         
    else:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(messenger+path)
        msg.setWindowTitle("Info")
        msg.exec_() 
        return False

            
#To evaluate vector readable     
def is_crs(crs1,crs2,messenger):
    
            if crs1 == crs2:
                return True
            
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(messenger+crs1+crs2)
                msg.setWindowTitle("Info")
                msg.exec_() 
                return False
            
#To evaluate field is integer     
def is_join(shape,messenger):
            if shape[0] ==0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(messenger+': '+str(shape))
                msg.setWindowTitle("Info")
                msg.exec_() 
                
                return False                
            
            else:
                             
                return True

