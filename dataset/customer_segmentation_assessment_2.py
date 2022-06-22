# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:35:30 2022

@author: End User
"""
import os
import pickle
import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras import Sequential,Input

from modules_for_customer import EDA,model_evaluation,functions

#%% Static
CSV_PATH = os.path.join(os.getcwd(),'Train.csv')
JOB_ENCODER_PATH = os.path.join(os.getcwd(),'job_encoder.pkl')
MARITAL_ENCODER_PATH = os.path.join(os.getcwd(),'marital_encoder.pkl')
EDUCATION_ENCODER_PATH = os.path.join(os.getcwd(),'education_encoder.pkl')
DEFAULT_ENCODER_PATH = os.path.join(os.getcwd(),'default_encoder.pkl')
HOUSING_ENCODER_PATH = os.path.join(os.getcwd(),'housing_encoder.pkl')
PERSONAL_ENCODER_PATH = os.path.join(os.getcwd(),'personal_encoder.pkl')
COMMUNICATION_ENCODER_PATH = os.path.join(os.getcwd(),'communication_encoder.pkl')
MONTH_ENCODER_PATH = os.path.join(os.getcwd(),'month_encoder.pkl')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'logs',log_dir)
#%% EDA
# Step 1) Data loading
df = pd.read_csv(CSV_PATH)

# Step 2) Data inspection
df.info()
df.describe().T
# 1) Presence of missing values
# 2) Able to analyze the categorical and continuous column

cat_columns = ['job_type','marital','education','default','housing_loan',
              'personal_loan','communication_type','month',
              'prev_campaign_outcome','term_deposit_subscribed']

con_columns = ['customer_age','balance','day_of_month','last_contact_duration',
               'num_contacts_in_campaign','days_since_prev_campaign_contact',
               'num_contacts_prev_campaign']

# Data visualization
# Plotting continuous columns and categorical columns
eda = EDA()
eda.data_visualization(df,con_columns,cat_columns)

# Comparing categorical data against term_deposite_subscribed.
# This helps to gain more insights about the dataset's
# relationship and understandings.
df.groupby(['term_deposit_subscribed','marital']).agg({'term_deposit_subscribed':'count'})
# But this method is not the best to display the comparison graph as it looks
# crowded and messy when we try to plot multiple columns against target columns.
# Its best when we want to analyze the comparison in detail up to knowing
# the values of comparisons.

# To plot categorical columns against target column.
eda.count_plot(df,cat_columns)
# This method is better as it displays the comparison clearer and more tidy.

# Step 3) Data cleaning
# convert categorical columns into integers
# convert target_column into OHE
# 1) copy original datasets into dummy datasets
df_dummy = df.copy()

# 2) check the presence of missing values
df_dummy.isna().sum()
# Missing values:
    # 1) customer_age
    # 2) marital
    # 3) balance
    # 4) personal_loan
    # 5) last_contact_duration
    # 6) num_contacts_in_campaign
    # 7) days_since_prev_campaign_contact

# 3) check duplicates
df_dummy.duplicated().sum()
# No presence of of duplicated datasets.

# 4) drop unrelated columns --> id column, prev_campaign_outcome column
df_dummy = df_dummy.drop(labels=['id','prev_campaign_outcome'],axis=1)

# 5) convert categorical columns into integers using LabelEncoder
# To check categorical columns required to be converted
df_dummy.dtypes
# The columns are:
    # job_type 
    # marital --> presence of nan
    # education
    # default
    # housing_loan
    # personal_loan --> presence of nan
    # communication_type
    # month

# 6) Since to convert categorical data into integers, labelencoder will 
    # overwrite NaN and this is wrong. Hence, we need to pass NaN while 
    # encoding those categorical data into int.
le = LabelEncoder()
# Create new categorical columns to exclude term_deposit_subscribed and
# prev_campaign_outcome.
cat_columns_new = ['job_type','marital','education','default','housing_loan',
                   'personal_loan','communication_type','month',]

paths = [JOB_ENCODER_PATH,MARITAL_ENCODER_PATH,EDUCATION_ENCODER_PATH,
         DEFAULT_ENCODER_PATH,HOUSING_ENCODER_PATH,PERSONAL_ENCODER_PATH,
         COMMUNICATION_ENCODER_PATH,MONTH_ENCODER_PATH]

# To generate PATH and save for pkl file for label encoding for categorical
# columns.
eda.create_pkl(cat_columns_new,df_dummy,paths)

# 7) The next step is to handle NaN.

knn_imp = KNNImputer()
df_dummy = knn_imp.fit_transform(df_dummy)
df_dummy = pd.DataFrame(df_dummy) # To convert array into dataframe
# To put back columns names
df_dummy.columns = df.drop(labels=['id','prev_campaign_outcome'],axis=1).columns
df_dummy.isna().sum() # No presence of NaNs

# To make sure no decimal places in categorical data

df.info()
df_dummy.info()
non_decimal_columns = ['job_type','marital','education','default',
                        'housing_loan','personal_loan','communication_type',
                        'day_of_month','month','num_contacts_prev_campaign',
                        'term_deposit_subscribed']
for i in non_decimal_columns:
    df_dummy[i] = np.floor(df_dummy[i])

# Step 4) Features selection
# 1) Regression Analysis
# Cramer's V --> categorical vs categorical
# To find the correlation of categorical data against term_deposit_subscribed.
func = functions()
for cat in cat_columns:
    if cat == 'prev_campaign_outcome':
        pass
    else:
        print(cat)
        confusion_mat = pd.crosstab(df_dummy[cat],df_dummy['term_deposit_subscribed']).to_numpy()
        print(func.cramers_corrected_stat(confusion_mat))

# 2) Logistic Regression
# To find correlation for continuous dataset against 
# categorical data (term_deposit_subscribed)
# make sure term_deposit_subscribed (y) is not in str.
# To print the correlation of continuous columns against target column.
eda.correlation_continuous_column(con_columns,df_dummy)

# 3) Take month (categorical data) since its achieved the highest correlation
    # among all at 0.27. The rest, take all continuous data as all of them
    # exceed 50% with accuracy of 89% in average.
X = df_dummy.loc[:,['month','customer_age','balance','day_of_month',
                    'last_contact_duration','num_contacts_in_campaign',
                    'days_since_prev_campaign_contact',
                    'num_contacts_prev_campaign']]
y = df_dummy['term_deposit_subscribed']
nb_class = len(np.unique(df_dummy['term_deposit_subscribed']))

# Step 5) Data preprocessing
# Features scalling
XSCALED_PATH = os.path.join(os.getcwd(),'xscaled.pkl')

ss = StandardScaler()
X_scaled = ss.fit_transform(X)
# Save scaler model
with open(XSCALED_PATH,'wb') as file:
    pickle.dump(ss,file)

#%% Deep learning
# OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=-1))
# Save ohe model
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,
                                                 test_size=0.3,
                                                 random_state=123)
# Model development
# 8 input shape
# 2 output shape
model = Sequential()
model.add(Input(shape=(8)))
model.add(Dense(32,activation='sigmoid',name='Hidden_layer_1'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(2,activation='sigmoid',name='Output_layer'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',
              metrics='acc')

# callbacks
tensorboard_callback = TensorBoard(log_dir=LOG_FOLDER_PATH)
early_stopping_callback = EarlyStopping(monitor='loss',patience=5)

# Model training
hist = model.fit(X_train,y_train,batch_size=128,epochs=10,
                 validation_data=(X_test,y_test),callbacks=tensorboard_callback)

hist.history.keys()
training_loss = hist.history['loss']
training_acc = hist.history['acc']
validation_acc = hist.history['val_acc']
validation_loss = hist.history['val_loss']

# Model evaluation
# To plot the graph for train_loss,val_loss,train_acc,val_acc
mod_evaluate = model_evaluation()
mod_evaluate.plot_graph(training_loss, validation_loss, training_acc, validation_acc)

results = model.evaluate(X_test,y_test)
print(results)

#%% Model saving
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
model.save(MODEL_SAVE_PATH)

#%% Plotting model architecture
# from tensorflow.keras.utils import plot_model
# plot_model(model,show_shapes=True,show_layer_names=(True))

#%% Discussion
# The model shows that it able to train for accuracy at 90% with loss at 24%.
# Model accuracy can be improved by adding more hidden layers.


