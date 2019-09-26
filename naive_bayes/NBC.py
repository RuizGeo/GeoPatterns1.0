#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:18:30 2018

@author: ruiz
"""
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qgis.core import *
import ogr
# Import the code for the dialog
from .nb_dialog import NBDialog
#import sys
import os
import numpy as np
from .to_evaluate import is_none, is_defined, exist_file,\
list_is_empty, txt_is_writable,field_is_integer,field_is_real,\
vector_is_readable,is_crs,is_join
#arg[1]=train_path
#arg[2]=dataset_path
#arg[3]=validation_path
#arg[4]=start_k
#arg[5]=end_k
#arg[6]=step_k
#arg[7]=field_class_train
#arg[8]=field_class_validation
#arg[9]=metrics_distance
#arg[10]=weigths_features
#arg[11]=state_applyclass
#arg[12]=assess_text_path
#arg[13]=output_classification

def model_NB(bar,path_train,dataset_path,\
                               path_val,field_class_train,\
                               field_class_val,a_priori,\
                               state_applyclass,assess_text_path,classification_path):
        try:        
            #Geodata mining
            from sklearn.naive_bayes import GaussianNB
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
            print('Import pandas')
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Error import pandas")
            msg.setWindowTitle("Info")
            msg.exec_() 
            return 0
        #To evaluate vector readable
        if vector_is_readable(path_train,'Error reading the ')==False:
            return 0
                    #Zero progressaBar
            
        bar.setValue(0)
        #create text
        f_txt=open(assess_text_path,'w')  
        #Write 
        f_txt.write('Dataset;Priori;kappa'+'\n')
        #get dataframe training samples
        #dft=gpd.read_file(path_train)
        train=QgsVectorLayer(path_train, 'train', 'ogr')
        #get dataframe validation samples
        #dfv=gpd.read_file(path_val)
        val=QgsVectorLayer(path_val, 'val', 'ogr')
        #Get CRS
        crsT=train.crs().authid()
        crsV=val.crs().authid()
        if is_crs (crsT,crsV,'CRS are different - (Training and validation samples)' )==False:
            return 0        
        #get names data set
        dataset_names=[f for f in os.listdir(dataset_path)  if f.endswith('.shp')]
        #best parameters
        best_parameters={'dataset':None}
        #acurcia
        acuracia=0.0
        #Len Segs names list
        len_segs = len(dataset_names)  
        #segmentations file
        for i_n, seg in enumerate(dataset_names):                
            #Selecionar arquivos .shp     
            #ProgressaBar
            bar.setValue(((i_n)/float(len_segs))*100)
            #f_txt.write(segs_path+os.sep+seg+'\n')
            print (dataset_path+os.sep+seg)
            if vector_is_readable(dataset_path+os.sep+seg,'Data set is not readable') == False:
                    return 0
            #Ler segmentacoes
            segmentation=QgsVectorLayer(dataset_path+os.sep+seg, 'seg', 'ogr')
            #get names columns fields - features
            features=segmentation.fields().names()
            #To evaluate CRS data set and samples
            crsS = segmentation.crs().authid()
            if is_crs (crsT,crsS,'CRS are different - (data set)' )==False:
                return 0
            #Ler segmentacoes
            #dfs=gpd.read_file(dataset_path+os.sep+seg)
            #create validation samples merge attribute spatial join
            #dfjv=gpd.sjoin(dfv,dfs,how="inner", op='intersects')
            dfjv=createSampleDF(dataset_path+os.sep+seg,path_val,field_class_val)
            #Criar amostras de treinamento, merge attribute spatial join
            #dfjt=gpd.sjoin(dft,dfs,how="inner", op='intersects')
            dfjt=createSampleDF(dataset_path+os.sep+seg,path_train,field_class_train)
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
            #Drop NaN validation
            dfjt=dfjt.dropna(subset=features)
            
            #To evatualte join
            if is_join(dfjt.shape,'Data set and training samples do not overlap or contains NaN') == False:
                return 0
            #Get features and remove geometry and id_seg
            #dfs.drop(['geometry','id_seg'],axis=1,inplace =True)
            #features=dfs.columns
            #Drop NaN validation
            dfjv=dfjv.dropna(subset=features)
            print('validation (rows, cols): ',dfjv.shape)
            print('training (rows, cols): ',dfjt.shape)
            #To evatualte join
            if is_join(dfjv.shape,'Data set and validation samples do not overlap or contains NaN') == False:
                return 0    
            #a priori prob
            freq=np.bincount(dfjt[field_class_train])
            freqs=freq/freq.sum()
            uniq=np.unique(dfjt[field_class_train])
            #apriori
            dic_a_priori={'False':None,'True':freqs[uniq]}
            #criar model NB
            clf = GaussianNB(priors=dic_a_priori[a_priori] )
            #Ajustar modelo
            modelNB = clf.fit(dfjt[features].values, dfjt[field_class_train])
            #Classificar
            clas = modelNB.predict(dfjv[features].values)
            #Calculate kappa
            kappa = metrics.cohen_kappa_score(dfjv[field_class_val],clas)                   
            #Calculate PC
            #pc,qd,qa,matrix=pontius2011(dfjv[field_class_val],clas)
            #print (pc,qd,qa)
            f_txt.write(seg+';'+a_priori+';'+str(round(kappa,4))+'\n') 
            #Avaliar a acuracia
            
            if kappa > acuracia:
                acuracia=kappa
                #Guardar parametros random forest
                best_parameters['dataset']=seg
        del(dfjv,dfjt)           
        #classificar segmentacao
        f_txt.write('############# Best Parameters #############'+'\n')
        f_txt.write('dataset: '+best_parameters['dataset']+'\n')
        ###################### classify best case##############################
        if bool(state_applyclass) :
            #Ler segmentacoes
            #df_dataset=gpd.read_file(dataset_path+os.sep+best_parameters['dataset'])
            df_dataset=createDF(dataset_path+os.sep+best_parameters['dataset'])
            #Remove NaN
            df_dataset=df_dataset.fillna(0)
            #create validation samples merge attribute spatial join
            #dfjv=gpd.sjoin(dfv,df_dataset,how="inner", op='intersects')
            dfjv=createSampleDF(dataset_path+os.sep+best_parameters['dataset'],path_val,field_class_val)
            #Criar amostras de treinamento, merge attribute spatial join
            #dfjt=gpd.sjoin(dft,df_dataset,how="inner", op='intersects')
            dfjt=createSampleDF(dataset_path+os.sep+best_parameters['dataset'],path_train,field_class_train)
            #a priori prob
            freq=np.bincount(dfjt[field_class_train])
            freqs=freq/freq.sum()
            uniq=np.unique(dfjt[field_class_train])
            #apriori
            dic_a_priori={'False':None,'True':freqs[uniq]}
            #criar model NB
            clf = GaussianNB(priors=dic_a_priori[a_priori] )            #criar modelo KNN
            
            #Ajustar modelo
            model = clf.fit(dfjt[features].values, dfjt[field_class_train])
            #Classificar
            clas = model.predict(dfjv[features].values)     
            #Calculate confusion matrix
            matrix=metrics.confusion_matrix(dfjv[field_class_val],clas)                 
            #Calculate PC
            #pc,qd,qa,matrix=pontius2011(dfjv[field_class_val],clas)
            #Remove NaN dat5aset
            df_dataset=df_dataset.dropna()

            
            #take values remove ID seg duplicate
            '''
            #Group by 
            #group_by=df_dataset.groupby(['id_seg']).mean()
            #Set index
            df_dataset.set_index('id_seg',inplace=True)
            for idx in group_by.index:
                if df_dataset.loc[idx].shape[0]!=df_dataset.loc[idx].size:
                    
                    #take values
                    df_dataset.loc[idx]=np.tile(group_by.loc[idx].values,(df_dataset.loc[idx].shape[0],1))
                else:
                     df_dataset.loc[idx]=group_by.loc[idx].values
                       #Reset index
            #Set index
            df_dataset.reset_index('id_seg',inplace=True)'''
                
            #classificar
            classification = model.predict(df_dataset[features].values)
            ##create aux DF classification
            df_dataset['target']=classification
            #output classification
 
            #output classification
            DFtoSHP(dataset_path+os.sep+best_parameters['dataset'],classification_path, df_dataset)
            #Write confusion matrix
            f_txt.write('############# Confusion Matrix #############'+'\n')
            f_txt.write(str(matrix.T)+'\n')
            #del
            del(df_dataset)
        else:
            pass
       
        #obter os melhores parametros
        #trees, max_d=
        print (best_parameters)
        #criar o modelo Random Forest
        #clf = ensemble.RandomForestClassifier( n_estimators =trees, max_depth =max_d,criterion=criterion_split)
        #Ajustar o modelo
        #modelTree = clf.fit(dfj[features].values, dfj[field_class_train])
        #Classificar segmentacao
        #dfs['classify']=modelTree.predict(dfs.values)
        #Calcular a probabilidade e converter para String
        #probs=[str(row) for row in np.round(modelTree.predict_proba(dfs[features].values)*100,2).tolist()]
        #insert geodata frame
        #dfs['probs']=probs
        #Save classify        
        #dfs[['geometry','classify','probs']].to_file(segs_path+os.sep+'class_'+seg)
        #Criar datafame com 
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
    #return
    return df.replace(NULL,np.nan)

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
    #return
    return df.replace(NULL,np.nan)        

def pontius2011(labels_validation,classifier):
        #get class
        labels = np.unique(labels_validation)
        #Get total class
        n_labels=labels.size        
        print ('n labels: ',n_labels)
        #create matrix 
        sample_matrix = np.zeros((n_labels,n_labels))
        #print sample_matrix
        #print np.count_nonzero(classifier==labels_validation)

        #Loop about labels
        for i,l in enumerate(labels):
            #Assess label in classifier
            selec=classifier==l
            print ( selec.any())
            if selec.any():
                #Get freqs
                coords,freqs=np.unique(labels_validation[selec],return_counts=True)
                print (coords,freqs)
                #insert sample_matrix
                sample_matrix[i,coords-1]=freqs
                print( 'l, Freqs: ',l,'---',freqs)
            
        print (sample_matrix)
        #Sample matrix: samples vs classification
        #sample_matrix=np.histogram2d(classifier,labels_validation,bins=(n_labels,n_labels))[0]
        #coo =np.array([4,5,8,9,11,12,13])-1
        #sample_matrix=sample_matrix[:,coo]
        #sample_matrix=sample_matrix[coo,:]
        print (sample_matrix.shape)
        #Sum rows sample matrix
        sample_total = np.sum(sample_matrix, axis=1)
        print ('sum rows: ',sample_total)
        #reshape sample total
        sample_total = sample_total.reshape(n_labels,1)
        #Population total: Image classification or labels validation (random)
        population = np.bincount(labels_validation)
        #Remove zero
        population = population[1:]
        print (population)
        
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
        #print 'Quantity disagreement :',quantity_disagreement
        #print 'Allocation disagreemen :',allocation_disagreemen
        #print 'Proportion correct: ',proportion_correct
        print ('PC: ',proportion_correct, ' DQ: ',quantity_disagreement, 'AD: ',allocation_disagreemen)
        return proportion_correct, quantity_disagreement, allocation_disagreemen,sample_matrix
    




