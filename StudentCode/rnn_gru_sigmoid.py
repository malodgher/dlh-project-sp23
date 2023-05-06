
#---------------------*********************************--------------------------------#
#---------------------*********************************--------------------------------#

# ***** Implemented by: Sagar dalwadi
# ***** Net ID: sagardd2

#---------------------*********************************--------------------------------#
#---------------------*********************************--------------------------------#

#!/usr/bin/env python
# coding: utf-8

#---------------------*********************************--------------------------------#
############# Install required libraries #############################
#---------------------*********************************--------------------------------#


#pip install pyhealth


#pip install pyreadr

#pip install pandas

#pip install sklearn

#pip install tensorflow


#---------------------*********************************--------------------------------#
################### Importing required modules/libraries ############################
#---------------------*********************************--------------------------------#

from pyhealth.datasets import MIMIC3Dataset
import pyreadr
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report


#---------------------*********************************--------------------------------#
######################## Testing with pyhealth dataset ###################
#---------------------*********************************--------------------------------#

# create MIMIC3Dataset object
mimic3base = MIMIC3Dataset(
    root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    # map all NDC codes to ATC 3-rd level codes in these tables
    code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
)


mimic3base.stat()

########################################################


#---------------------*********************************--------------------------------#
################### Input data preprocessing ############################
#---------------------*********************************--------------------------------#



data_dir = "/dlh-project-sp23/StudentCode/InputData/"
inputdf_records = 15000
max_epoch = 3

# load the DIAGNOSES_ICD table as a Pandas dataframe
diagnoses_icd_df = pd.read_csv(data_dir + "DIAGNOSES_ICD.csv")

diagnoses_icd_df.head()

# get the statistics of the dataframe
diagnoses_icd_stats = diagnoses_icd_df.describe()

# print the statistics
print(diagnoses_icd_stats)

# load the PROCEDURES_ICD table as a Pandas dataframe
procedures_icd_df = pd.read_csv(data_dir + "PROCEDURES_ICD.csv")

# load the PRESCRIPTIONS table as a Pandas dataframe
prescriptions_df = pd.read_csv(data_dir + "PRESCRIPTIONS.csv")


# get the statistics of the dataframe
procedures_icd_stats = procedures_icd_df.describe()

# print the statistics
print(procedures_icd_stats)

# get the statistics of the dataframe
prescriptions_stats = prescriptions_df.describe()

# print the statistics
print(prescriptions_stats)

# load the ADMISSIONS table as a Pandas dataframe
admissions_df = pd.read_csv(data_dir + "ADMISSIONS.csv")


# get the statistics of the dataframe
admissions_stats = admissions_df.describe()

# print the statistics
print(admissions_stats)
admissions_df.head()


# load the ICUSTAYS table as a Pandas dataframe
icustays_df = pd.read_csv(data_dir + "ICUSTAYS.csv")

# get the statistics of the dataframe
icustays_stats = icustays_df.describe()

# print the statistics
print(icustays_stats)

# load the PATIENTS table as a Pandas dataframe
patients_df = pd.read_csv(data_dir + "PATIENTS.csv")

# get the statistics of the dataframe
patients_stats = patients_df.describe()

# print the statistics
print(patients_stats)

# merge the tables together based on the common patient identifier
patient_admissions = pd.merge(admissions_df, patients_df, on="SUBJECT_ID")
patient_admissions_icd = pd.merge(patient_admissions, diagnoses_icd_df, on=["SUBJECT_ID", "HADM_ID"])
patient_admissions_icustays = pd.merge(patient_admissions_icd, icustays_df, on=["SUBJECT_ID", "HADM_ID"])

# print the resulting dataframe
print(patient_admissions_icustays.head())

# Define heart failure ICD codes
hf_codes = ['4280', '4281', '4289','42820','42821','42822','42823','42830','42831','42832','42833']

# Create new column for heart failure flag
patient_admissions_icustays['heart_failure'] = np.where(patient_admissions_icustays['ICD9_CODE'].isin(hf_codes) , 1, 0)

patient_admissions_icustays.head()


columns = patient_admissions_icustays.columns.tolist()
print(columns)

input_df = patient_admissions_icustays[['SUBJECT_ID','MARITAL_STATUS','DIAGNOSIS','GENDER','ICD9_CODE','LOS','heart_failure']]

input_df.head()
input_onehot_df = input_df

input_onehot_df.head()
input_onehot_df = input_onehot_df.drop_duplicates()


#---------------------*********************************--------------------------------#
################### One hot encoding ############################
#---------------------*********************************--------------------------------#




one_hot = pd.get_dummies(input_onehot_df['MARITAL_STATUS'])
input_onehot_df = input_onehot_df.drop('MARITAL_STATUS', axis=1)
input_onehot_df = input_onehot_df.join(one_hot)

one_hot = pd.get_dummies(input_onehot_df['DIAGNOSIS'])
input_onehot_df = input_onehot_df.drop('DIAGNOSIS', axis=1)
input_onehot_df = input_onehot_df.join(one_hot)

one_hot = pd.get_dummies(input_onehot_df['GENDER'])
input_onehot_df = input_onehot_df.drop('GENDER', axis=1)
input_onehot_df = input_onehot_df.join(one_hot)


input_onehot_df = input_onehot_df.drop('ICD9_CODE', axis=1)
#input_onehot_df = input_onehot_df.join(one_hot)

input_onehot_df = input_onehot_df.drop('LOS', axis=1)
input_onehot_df = input_onehot_df.drop('SUBJECT_ID', axis=1)

input_onehot_df.head()
input_limited_df = input_onehot_df.loc[:inputdf_records, :]



#---------------------*********************************--------------------------------#
################### RNN model implementation ############################
#---------------------*********************************--------------------------------#




# Split the data into input (X) and target (y) variables
X = input_limited_df.drop(['heart_failure'], axis=1)
y = input_limited_df['heart_failure']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



X_train.astype('float32')
X_test.astype('float32')

# Define the number of features and the length of the input sequences
num_features = X_train.shape[1]
seq_length = 1

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=2, output_dim=num_features, input_length=seq_length))
model.add(GRU(units=32, input_shape=(seq_length, num_features)))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

# Define the model architecture
#model = Sequential()
#model.add(Embedding(input_dim=2, output_dim=num_features, input_length=seq_length))
#model.add(GRU(units=32, activation='relu', input_shape=(seq_length, num_features)))
#model.add(Dense(units=16, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
optimizer = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#model.summary()

# Fit the model to the training data
#history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train.values.reshape(-1, 1, num_features), y_train.values, epochs=3, batch_size=5, validation_data=(X_test.values.reshape(-1, 1, num_features), y_test.values))

model.fit(X_train, y_train, epochs=max_epoch, batch_size=5, validation_data=(X_test, y_test))


# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)

# Get the model predictions on the testing set
y_pred = model.predict(X_test)

# Convert predicted probabilities to class labels
y_pred_labels = np.round(y_pred)

# Print classification report
print(classification_report(y_test, y_pred_labels, zero_division=1))



