# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:57:31 2022

@author: End User
"""

import pickle
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

class functions():
    def __init__(self):
        pass
    
    def cramers_corrected_stat(self,confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

class EDA():
    def __init__(self):
        pass
    
    def data_visualization(self,df,con_columns,cat_columns):
        for con in con_columns:
            plt.figure()
            sns.distplot(df[con])
            plt.show()
        for cat in cat_columns:
            plt.figure(figsize=(20,20))
            sns.countplot(df[cat])
            plt.show()
    def count_plot(self,df,cat_columns):
        for i in cat_columns:
            plt.figure()
            sns.countplot(df[i],hue=df['term_deposit_subscribed'])
            plt.show()
    def create_pkl(self,cat_columns_new,df_dummy,paths):
        for index,i in enumerate(cat_columns_new):
            # print(index)
            le = LabelEncoder()
            temp = df_dummy[i]
            temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()])
            df_dummy[i] = pd.to_numeric(temp,errors='coerce')
            with open(paths[index],'wb') as file:
                pickle.dump(le,file)
    def correlation_continuous_column(self,con_columns,df_dummy):
        for con in con_columns:
            print(con)
            lr = LogisticRegression()
            lr.fit(np.expand_dims(df_dummy[con],axis=-1),df_dummy['term_deposit_subscribed'])
            print(lr.score(np.expand_dims(df_dummy[con],axis=-1),df_dummy['term_deposit_subscribed']))
class model_evaluation():
    def __init__(self):
        pass
    def plot_graph(self,training_loss,validation_loss,training_acc,validation_acc):
        plt.figure()
        plt.plot(training_loss)
        plt.plot(validation_loss)
        plt.legend(['train_loss','val_loss'])
        plt.show()

        plt.figure()
        plt.plot(training_acc)
        plt.plot(validation_acc)
        plt.legend(['train_acc','val_acc'])
        plt.show()


