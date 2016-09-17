# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:35:51 2016

@author: rahma.chaabouni
"""
from __future__ import division

import os
import theano.tensor as T
import numpy as np
import time
import pandas as pd
import theano

os.chdir('/home/spark/Documents/Recherches/JouerSurLesParam/LasagneToTheano/')
from logistic_sgd import load_data
from AdaM2 import AdaBoost_M2, Hypothesis
from Plot_Digits import displayimage


os.chdir('/home/spark/Documents/Recherches/Theano/Datasets')
dataset='mnist.pkl.gz'
datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
val_set_x, val_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

x_train = train_set_x.get_value(borrow = True)
Y_true_train = train_set_y.eval()

x_test = test_set_x.get_value(borrow = True)
Y_true_test = test_set_y.eval()

#%%
T_max = (1,7000)
#T_max = (10,100,2000)

er_train = []
er_test = []
# allocate symbolic variables for the data
t0 = time.time()
for t in T_max:
    H = AdaBoost_M2(train_set_x,train_set_y, t ,reg_L1 = 0.0, reg_L2 = 0.0, learning_rate=0.01,
             n_epochs=29, batch_size=20, n_in = 28*28, 
             n_hiddens = [10, 10], n_out = 10, 
             activations = [T.nnet.relu, T.nnet.relu], type_grad = 'nesterov')
    t1 = time.time()
    print('temps pour AdaBoostM2', t1 - t0)
    
    
    #predictions = h.predict(train_set_x.get_value())
    #
    #y_true_train = train_set_y.eval()
    #CM_train = confusion_matrix(y_true_train, predictions)
    #plot = sns.heatmap(pd.DataFrame(CM_train), annot=True, fmt="d", linewidths=.5)
    #fig1 = plot.get_figure()
    #
    #print(CM_train)
    #print(compute_error_confMatrix(CM_train, len(y_true_train)))
    
    #Evaluation
    #apprentissage             
    predictions_train = Hypothesis(H[0], H[1], x_train)
    erreur_train = np.sum(predictions_train != Y_true_train)/len(Y_true_train)
    er_train.append(erreur_train)
    
    #test
    predictions_test = Hypothesis(H[0], H[1], x_test)
    erreur_test = np.sum(predictions_test != Y_true_test)/len(Y_true_test)
    er_test.append(erreur_test)

    
#%% 
"""
Pour analyser et sauvegarder les résultats de l'apprentissage.
La variable "changement" est utilisée pour nommer les fichiers sauvegardés.        
"""    
########################################################################
###                    Changement par iteration                       ##
########################################################################
#changement = '10_2_relu_optimisation_V1_7000'
#
########################################################################
###               Sauvegarder les resultats des erreurs               ##
########################################################################
#df_erreur = pd.DataFrame({'train': er_train, 'test': er_test})
#path_erreur = '/home/spark/Documents/Recherches/Rapport/Resultat_AdaM2/'+ 'erreur_tanh_' + changement
#df_erreur.to_csv(path_erreur, index = False)
#
########################################################################
###                 afficher les digits mal classées                  ##
########################################################################
#
#false_pos_train = np.where(predictions_train != Y_true_train)[0]
#false_label_train = [predictions_train[x] for x in false_pos_train]
#df_train = pd.DataFrame({'pos': false_pos_train, 'label': false_label_train})
#path_1 = '/home/spark/Documents/Recherches/Rapport/Resultat_AdaM2/'+ 'digit_tanh_' + changement
#df_train.to_csv(path_1, index = False)
## df_train = pd.read_csv(path_1)
#
#false_pos_test = np.where(predictions_test != Y_true_test)[0]
#false_label_test = [predictions_test[x] for x in false_pos_test]
#df_test = pd.DataFrame({'pos': false_pos_test, 'label': false_label_test})
#path_2 = '/home/spark/Documents/Recherches/Rapport/Resultat_AdaM2/'+ 'digit_tanh_' + changement + '_test'
#df_test.to_csv(path_2, index = False)
## df_train = pd.read_csv(path_2)
#
########################################################################
###                 afficher les matrices de confusion                ##
########################################################################
#from sklearn.metrics import confusion_matrix
#print('on est la')
#string_1 = '/home/spark/Documents/Recherches/Rapport/Resultat_AdaM2/'+"heatmap_tanh_"+changement+ '.csv'
#string_2 = '/home/spark/Documents/Recherches/Rapport/Resultat_AdaM2/'+ "heatmap_tanh_"+changement+"_test.csv"
#
#CM_train = confusion_matrix(Y_true_train, predictions_train)
#np.savetxt(string_1, CM_train)
#
#CM_test = confusion_matrix(Y_true_test, predictions_test)
#np.savetxt(string_2, CM_test)
#
########################################################################
###                              HeatMap                             ##
########################################################################
##
##import seaborn as sns
##
##The_plot = sns.heatmap(pd.DataFrame(CM_train), annot= True, fmt = "d", linewidths=.5, vmax =100)
##fig = The_plot.get_figure()
###fig.savefig(string_1)
##
##The_plot = sns.heatmap(pd.DataFrame(CM_test), annot= True, fmt = "d", linewidths=.5)
##fig = The_plot.get_figure()
###fig.savefig(string_2)
##
#########################################################################
####                            Plot Digits                            ##
#########################################################################
##
##false_train = np.where(predictions_train != Y_true_train)[0]
##displayimage(false_train, x_train, Y_true_train, 
##               type_donnees = 'numpy', pred = predictions_train)
##
##
##false_test = np.where(predictions_test != Y_true_test)[0]
##displayimage(false_test, x_test, Y_true_test, 
##               type_donnees = 'numpy', pred = predictions_test)
##
#
