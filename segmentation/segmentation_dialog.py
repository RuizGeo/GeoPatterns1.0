# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SegmentationDialog
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

import os
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtWidgets
from .segmentation_ui import Ui_Dialog


#FORM_CLASS, _ = uic.loadUiType(os.path.join(
#    os.path.dirname(__file__), 'segmentation_dialog_base.ui'))


class SegmentationDialog(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        # Set up the user interface from Designer.
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        