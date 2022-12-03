#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:02:43 2018

@author: root
"""
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qgis.core import *
import ogr
# Import the code for the dialog
from .nb_dialog import NBDialog
import sys
import os
import numpy as np
from .to_evaluate import is_none, is_defined, exist_file,\
list_is_empty, txt_is_writable,field_is_integer,field_is_real,\
vector_is_readable,is_crs,is_join

 
    
def NB(bar,path_train,segs_path,\
                                   path_val,start_est,\
                                   end_est,step_est,start_dp,\
                                   end_dp,step_dp,field_model_train,\
                                   field_model_val,criterion_split,path_assess_file,stateCheckBox,model_path,type_model):
        

            try:        
                #Geodata mining
                from sklearn import tree
                from sklearn import metrics
                print('Import scikit-learn')
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Error import scikit-learn")
                msg.setWindowTitle("Info")
                msg.exec_() 
                return 0
            try:        
                #Geodata mining
                import pandas as pd
                global pd
                import rtree
                print('Import pandas')
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Error import pandas or rtree")
                msg.setWindowTitle("Info")
                msg.exec_() 
                return 0
            #To evaluate vector readable
            if vector_is_readable(path_train,'Error reading the ')==False:
                return 0

            #Zero progressaBar
            
            bar.setValue(0)
            #get dataframe training samples
            train=QgsVectorLayer(path_train, 'train', 'ogr')
            #dft=gpd.read_file(path_train)
        
            #get dataframe validation samples
            #dfv=gpd.read_file(path_val)
            val=QgsVectorLayer(path_val, 'val', 'ogr')
            #Get CRS
            crsT=train.crs().authid()
            crsV=val.crs().authid()
            if is_crs (crsT,crsV,'CRS are different - (Training and validation samples)' )==False:
                return 0
            #get names segmentations
            segs_names=[f for f in os.listdir(segs_path)  if f.endswith('.shp')]
            #best parameters
            best_parameters={'MinSam':0,'Depth':0}
            #acurcia
            acuracia_clas=0.0
            acuracia_regress=10000
            #create text
            f_txt=open(path_assess_file,'w')
            

            #To evaluaate if is model type
            if type_model =='classification':
               #Write 
               f_txt.write('Dataset;MinSam;Depth;kappa'+'\n')
               
            else:
                
                #Write 
                f_txt.write('Dataset;MinSam;Depth;MSE'+'\n') 
            #Len Segs names list
            len_segs = len(segs_names)        
            #segmentations file
            for i_n, seg in enumerate (segs_names): 
                #ProgressaBar
                bar.setValue(((i_n+1)/float(len_segs))*100)
                #Selecionar arquivos .shp          
                #f_txt.write(segs_path+os.sep+seg+'\n')
                print (segs_path+os.sep+seg)
                if vector_is_readable(segs_path,'Data set is not readable') == False:
                    return 0
                #Ler segmentacoes
                segmentation=QgsVectorLayer(segs_path+os.sep+seg, 'seg', 'ogr')
                #get names columns fields - features
                features=segmentation.fields().names()
                #To evaluate CRS data set and samples
                crsS = segmentation.crs().authid()
                if is_crs (crsT,crsS,'CRS are different - (data set)' )==False:
                    return 0
                #Create DatFrame segmentation
                #dfs=createDF(segs_path+os.sep+seg)
                
                #create validation samples merge attribute spatial join
                #dfjv=gpd.sjoin(dfv,dfs,how="inner", op='intersects')
                dfjv=createSampleDF(segs_path+os.sep+seg,path_val,field_model_val)

                #Criar amostras de treinamento, merge attribute spatial join
                #dfjt=gpd.sjoin(dft,dfs,how="inner", op='intersects')
                dfjt=createSampleDF(segs_path+os.sep+seg,path_train,field_model_train)
                #Get features and remove geometry and id_seg
                if 'id_seg' in features:
                    #Remover column
                    features.remove('id_seg')
                    #remove duplicates validation
                    #dfjv=dfjv.drop_duplicates(subset='id_seg')
                    #remove duplicates training
                    #dfjt=dfjt.drop_duplicates(subset='id_seg')
                else:
                    pass
                #Get columns names equal dtype=float
                #features=dfjt.select_dtypes(include='float').columns                
                #features=dfjt.columns  
                #Drop NaN validation
                dfjt=dfjt.dropna(subset=features)
                
                #To evatualte join
                if is_join(dfjt.shape,'Data set and training samples do not overlap or contains NaN') == False:
                    return 0
                #Drop NaN validation
                dfjv=dfjv.dropna(subset=features)
                print('validation (rows, cols): ',dfjv.shape)
                print('training (rows, cols): ',dfjt.shape)
                #To evatualte join
                if is_join(dfjv.shape,'Data set and validation samples do not overlap or contains NaN') == False:
                    return 0    


                #To evaluaate if is model type
                if type_model =='classification':
                    #Avaliar parametros da segmentacao
                    for t in range(int(start_est),int(end_est)+int(step_est),int(step_est)):
                        #Set progressbar
                        for md in range(int(start_dp),int(end_dp)+int(step_dp),int(step_dp)):  
                            
                            #criar modelo Random
                            clf = tree.DecisionTreeClassifier( criterion=criterion_split, max_depth =md, min_samples_split =t )
                            #Ajustar modelo
                            modelTree = clf.fit(dfjt[features].values, dfjt[field_model_train].values.astype(np.int8))
                            #Classificar
                            clas = modelTree.predict(dfjv[features].values)
                            #Calculate kappa
                            kappa = metrics.cohen_kappa_score(dfjv[field_model_val],clas)
                            #Calculate PC
                            #pc,qd,qa,matrix=self.pontius2011(dfjv[field_model_val],clas)
                            #print (pc,qd,qa)
                            f_txt.write(seg+';'+str(t)+';'+ str(md)+';'+str(round(kappa,4))+'\n') 
                            #Avaliar a acuracia
                            #print('Acc: '+str(acuracia)+' Pc: '+str(pc))
                            if kappa > acuracia_clas:
                                acuracia_clas=kappa
                                #Guardar parametros random forest
                                
                                best_parameters['MinSam']=t
                                best_parameters['Depth']=md
                                best_parameters['Dataset']=seg
                else:
                    #Avaliar parametros da segmentacao
                    for t in range(int(start_est),int(end_est)+int(step_est),int(step_est)):
                        #Set progressbar
                        for md in range(int(start_dp),int(end_dp)+int(step_dp),int(step_dp)):
                            #criar modelo Random Forest
                            clf = tree.DecisionTreeRegressor( criterion=criterion_split, max_depth =md, min_samples_split =t )
                            #Ajustar modelo
                            modelTree = clf.fit(dfjt[features].values, dfjt[field_model_train].values.astype(np.float32))
                            #Classificar
                            regress = modelTree.predict(dfjv[features].values)
                            #Calculate AUC
                            #auc =metrics.roc_auc_score(dfjv[field_model_val],regress)
                            mse=metrics.mean_squared_error(dfjv[field_model_val],regress)
                            #print (pc,qd,qa)
                            f_txt.write(seg+';'+str(t)+';'+ str(md)+';'+ str(round(mse,4))+'\n') 
                            #Avaliar a acuracia
                            #print('Acc: '+str(acuracia)+' Pc: '+str(pc))
                            if mse < acuracia_regress:
                                acuracia_regress=mse
                                #Guardar parametros random forest
                                
                                best_parameters['MinSam']=t
                                best_parameters['Depth']=md
                                best_parameters['Dataset']=seg
 

            #del dataframes
            del(dfjv,dfjt)   

            #classificar segmentacao
            f_txt.write('############# Best Parameters #############'+'\n')
            f_txt.write('Data set: '+best_parameters['Dataset']+' - '+'MinSam: '+str(best_parameters['MinSam'])+ ' - Depth:'+str(best_parameters['Depth'])+'\n')
            out_tree=path_assess_file.replace('.txt','.dot')
            tree.export_graphviz(modelTree,out_tree)
            ###################### classify best case##############################
            if eval(stateCheckBox) :
                #Ler segmentacoes
                #df_dataset=gpd.read_file(segs_path+os.sep+best_parameters['Dataset'])
                df_dataset=createDF(segs_path+os.sep+best_parameters['Dataset'])
                #Remove NaN
                df_dataset=df_dataset.fillna(0)
                #create validation samples merge attribute spatial join
                #dfjv=gpd.sjoin(dfv,dfs,how="inner", op='intersects')
                dfjv=createSampleDF(segs_path+os.sep+best_parameters['Dataset'],path_val,field_model_val)

                #Criar amostras de treinamento, merge attribute spatial join
                #dfjt=gpd.sjoin(dft,dfs,how="inner", op='intersects')
                dfjt=createSampleDF(segs_path+os.sep+best_parameters['Dataset'],path_train,field_model_train)
                #create validation samples merge attribute spatial join
                #dfjv=gpd.sjoin(dfv,df_dataset,how="inner", op='intersects')
                #Criar amostras de treinamento, merge attribute spatial join
                #dfjt=gpd.sjoin(dft,df_dataset,how="inner", op='intersects')
                #Drop NaN validation
                dfjt=dfjt.dropna(subset=features)
                #Drop NaN validation
                dfjv=dfjv.dropna(subset=features)
                #Get features and remove geometry and id_seg
                '''if 'id_seg' in df_dataset.columns:                    
                    #remove duplicates validation
                    #dfjv=dfjv.drop_duplicates(subset='id_seg')
                    #remove duplicates training
                    #dfjt=dfjt.drop_duplicates(subset='id_seg')'''
                #Apply model
                if type_model =='classification':
                    #criar modelo CART11
                    clf = tree.DecisionTreeClassifier( criterion=criterion_split, max_depth =best_parameters['Depth'], min_samples_split =best_parameters['MinSam'] )
                    #Ajustar modelo
                    model = clf.fit(dfjt[features].values, dfjt[field_model_train].values.astype(np.int8))
                    #Classificar
                    clas = model.predict(dfjv[features].values)                      
                    #Calculate confusion matrix
                    matrix=metrics.confusion_matrix(dfjv[field_model_val],clas)
                    
                        
                    #Classificar
                    classification = modelTree.predict(df_dataset[features].values)
                    ##create aux DF classification
                    df_dataset['target']=classification
                    #output classification
                    DFtoSHP(segs_path+os.sep+best_parameters['Dataset'],model_path, df_dataset)
                    #df_dataset[['geometry','classes']].to_file( model_path)
                    f_txt.write('############# Confusion Matrix #############'+'\n')
                    f_txt.write(str(matrix.T)+'\n')
                else:
                    #Create CART Regressor
                    clf = tree.DecisionTreeRegressor( criterion=criterion_split, max_depth =best_parameters['Depth'], min_samples_split =best_parameters['MinSam'] )

                    #Ajustar modelo
                    model = clf.fit(dfjt[features].values, dfjt[field_model_train].values.astype(np.float32))
                    #Regressor
                    regress = model.predict(df_dataset[features].values)   
                    ##create aux DF classification
                    df_dataset['target']=regress 
                    #output regression
                    DFtoSHP(segs_path+os.sep+best_parameters['Dataset'],model_path, df_dataset)
                    #output classification
                    #df_dataset[['geometry','values']].to_file( model_path)
                #Write text
                f_txt.write('############# Features #############'+'\n')
                f_txt.write(str(features)+'\n')
                f_txt.write('############# Features Importances #############'+'\n')
                feat_importances = np.around(model.feature_importances_*100,2)
                f_txt.write(str(feat_importances.tolist())+'\n')  

                #del
                del(df_dataset)
            else:
                pass
    
            print ('Best parameters: ',best_parameters)

            #close text       
            f_txt.close()
            
def DFtoSHP(shp_input, shp_out, dataframe):

    #Read shapefile imput
    layer = QgsVectorLayer(shp_input, 'imput', 'ogr') 
    count=layer.featureCount()
    
    # define fields for feature attributes. A QgsFields object is needed
    fields = QgsFields()
    fields.append(QgsField('target', QVariant.Double, 'double', 20, 3))
    #writer shapefiles
    writer = QgsVectorFileWriter(shp_out, "utf-8", fields, ogr.wkbPolygon, layer.crs(), "ESRI Shapefile")
   
    # add a feature
    fet = QgsFeature()
    #Get features input
    features= layer.getFeatures()
    #Loop features input    
    for feat in features:
        #set geometry
        fet.setGeometry(feat.geometry())
        #Set attributes
        #print(dataframe['classes'][feat.id()])
        fet.setAttributes([float(dataframe['target'][feat.id()])])
        #Writer features
        writer.addFeature(fet)
    
    del writer

    
def createDF(dataset):

    #Read shapefile segmentation
    layer_seg = QgsVectorLayer(dataset, 'dataset', 'ogr')
    #counts polygons
    counts=layer_seg.featureCount()
    #dtypes fields 
    fields=[f for f in layer_seg.fields().names()]
    
    #d_types.append((field_class,np.object))
    #Creat numpy array to store datas
    datas=np.zeros(shape=(counts,len(fields)),dtype=np.object) 
    idxs=np.zeros(shape=(counts,1),dtype=np.object) 
    #Get features segmentacao
    features_seg = layer_seg.getFeatures()
    #Loop reference data
    for feat in features_seg:          
        #insert values 
        datas[feat.id()]=feat.attributes()
        idxs[feat.id()]=feat.id()
    #insert fileds idxs
    fields.append('idxs')
    #Concatenar and create DataFrame
    df=pd.DataFrame(np.concatenate((datas, idxs), axis=1), columns=fields)
    #Remove NULL
    df[df==NULL] = np.nan
    #return
    return df#.replace(NULL,np.nan)

def createSampleDF(dataset,sample,field_class):

    
    #Read training and validation
    sample=QgsVectorLayer(sample, 'sample', 'ogr')

    #Loop about segmentations

    #Read shapefile segmentation
    layer_seg = QgsVectorLayer(dataset, 'dataset', 'ogr')
    #counts polygons
    counts_samples=sample.featureCount()
    #dtypes fields 
    fields=[f for f in layer_seg.fields().names()]
    #d_types.append((field_class,np.object))
    #Creat numpy array to store datas
    datas=np.zeros(shape=(counts_samples,len(fields)),dtype=np.object)
    classes=np.zeros(shape=(counts_samples,1),dtype=np.object)
    #Get features reference data
    sample_feats = sample.getFeatures()
  
    #Get features segmentacao
    features_seg = layer_seg.getFeatures()
    #Create spatial index
    index = QgsSpatialIndex()
    for feats_t in sample_feats:
        index.insertFeature(feats_t)

    #Loop reference data
    for feat in features_seg:          
        #Obter os ids pela interseccao entre obj e val
        ids=index.intersects(feat.geometry().boundingBox())
        #if polygons intersects points samples
        if ids:
            #Selecionar o objs com ids
            sample.selectByIds(ids)
            #features selecionadas
            features_selec = sample.selectedFeatures()
            #percorrer features selecionadas
            for sel in features_selec:
                #if polygon contains points
                if feat.geometry().contains(sel.geometry()):
                    
                    #insert values 
                    datas[sel.id()]=feat.attributes()
                    classes[sel.id()]=sel[field_class]
                        
    #Fields                   
    fields.append(field_class)
    #Concatenar and create DataFrame
    df=pd.DataFrame(np.concatenate((datas, classes), axis=1), columns=fields)
    #Remove NULL
    df[df==NULL] = np.nan
    #return
    return df#.replace(NULL,np.nan)


    
def pontius2011(labels_validation,classifier):
            #get class
            labels = np.unique(labels_validation)
            #Get total class
            n_labels=labels.size        

            #create matrix 
            sample_matrix = np.zeros((n_labels,n_labels))

            #Loop about labels
            for i,l in enumerate(labels):
                #Assess label in classifier
                selec=classifier==l
                #print ( selec.any())
                if selec.any():
                    #Get freqs
                    coords,freqs=np.unique(labels_validation[selec],return_counts=True)
                    #print (coords,freqs)
                    #insert sample_matrix
                    sample_matrix[i,coords-1]=freqs
                    #print( 'l, Freqs: ',l,'---',freqs)
                

            #Sum rows sample matrix
            sample_total = np.sum(sample_matrix, axis=1)
            #print ('sum rows: ',sample_total)
            #reshape sample total
            sample_total = sample_total.reshape(n_labels,1)
            #Population total: Image classification or labels validation (random)
            population = np.bincount(labels_validation)
            #Remove zero
            population = population[1:]
            #print (population)
            
            #population matrix
            pop_matrix = np.multiply(np.divide(sample_matrix,sample_total,where=sample_total!=0),(population.astype(float)/population.sum()))
            
            #comparison total: Sum rows pop_matrix
            comparison_total = np.sum(pop_matrix, axis=1)
            #reference total: Sum columns pop_matrix
            reference_total = np.sum(pop_matrix, axis=0)
            #overall quantity disagreement
            quantity_disagreement=(abs(reference_total-comparison_total).sum())/2.
            #overall allocation disagreemen
            dig =pop_matrix.diagonal()
            comp_ref=np.dstack((comparison_total-dig,reference_total-dig))
            #allocation_disagreemen=((2*np.min(comp_ref,-1)).sum())/2.
            #proportion correct
            proportion_correct = np.trace(pop_matrix)
            allocation_disagreemen=1-(proportion_correct+quantity_disagreement)

            print ('PC: ',proportion_correct, ' DQ: ',quantity_disagreement, 'AD: ',allocation_disagreemen)
            return proportion_correct, quantity_disagreement, allocation_disagreemen, sample_matrix
            
