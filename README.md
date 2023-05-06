# dlh-project-sp23
CS598 Deep Learning for Healthcare Spring 2023 project code by Murtaza Lodgher and Sagar Dalwadi

There are 2 sections: 1) Used Author's code 2) Implemented our own code

### *****************************************************************************

# Author's Code
#### (Under Authors code folder)

The first step of our implementation involved using the original code base provided in the paper and making the necessary updates to the code to make it compatible with the latest version of Python including necessary updates to the libraries used such that we are able to successfully run the model against the data and produce the results.

We also utilized the synthetic data provided and passed it through the model for testing purposes and successfully generated the AUC for each epoch.

#### Heart Failure Prediction using RNN
This is a simple RNN (implemented with Gated Recurrent Units) for predicting a HF diagnosis given patient records.

There are four different versions:

1. gru_onehot.py: This uses one-hot encoding for the medical code embedding
2. gru_onehot_time.py: This uses one-hot encoding for the medical code embedding. This uses time information in addition to the code sequences
3. gru_emb.py: This uses pre-trained medical code embeddings. 
4. gru_emb_time.py: This uses pre-trained medical code embeddings. This suses time information in addition to the code sequences.

The data are synthetic and make no sense at all. It is intended only for testing the codes.

1. sequences.pkl: This is a pickled list of list of integers. Each integer is assumed to be some medical code.
2. times.pkl: This is a pickled list of list of integers. Each integer is assumed to the time at which the medical code occurred.
3. labels.pkl: This is a pickled list of 0 and 1s.
4. emb.pkl: This is a randomly generated code embedding of size 100 X 100

#### Requirement
Python and Theano are required to run the scripts

#### How to Execute
1. python gru_onehot.py sequences.pkl labels.pkl <output>
2. python gru_onehot_time.py sequences.pkl times.pkl labels.pkl <output>
3. python gru_emb.py sequences.pkl labels.pkl emb.pkl <output>
4. python gru_emb_time.py sequences.pkl times.pkl labels.pkl emb.pkl <output>

#### Reference
https://github.com/mp2893/rnn_predict

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5391725/pdf/ocw112.pdf


### ******************************************************************************

# Student's Code (Implemented our own)
#### (Under Student code folder)
#### Implemented by Sagar Dalwadi - sagardd2
#### Corresponding Jupyter Notebook is provided: CS598_DLH_Project_Sagar.ipynb

Now that we experimented with Author's code and got an understanding of how the model works & what data should be used, I implemented my own code with python keras library and real world data from PhysioNet website.

# Heart failure detection model usig RNN
Similar to the model provided in paper, this is a simple RNN with GRU for predicting heart failure for patient records.

There are 2 different versions:

1. rnn_gru_sigmoid.py: This uses sigmoid as activation function as part of RNN with GRU model. This uses admissions, patients, diagnoses, icu stays data from PhysioNet
2. rnn_gru_relu.py: This implements one of the ablations we proposed and uses ReLu activation function as part of RNN with GRU model. In addition to dataset mentioned in 1st version, it also uses lab events data for patients.

# Data

Data sets are extracted from PhysioNet website: (you need to have "CITI Data or Specimens Only Research" training completed & approved before you can access the data)
https://physionet.org/content/mimiciii/1.4/

#### Note: For the purpose of this project submission & to be able to allow TAs/Instructor to run the code successfully, I have made the data files used in the code accessible on google drive. Also, uploaded the smaller files to github under InputData folder for reference.

Once you have access go to Files section on above link where you can download required data files.

1. Admissions: gives information regarding a patient’s admission to the hospital
2. Patients: Provides information on each patient
3. Diagnoses ICD: Contains ICD diagnoses for patients
4. ICU Stays: defines a single ICU stay
5. Lab events: Contains all laboratory measurements for a given patient

More information on dataset, fields etc can be found at: https://mimic.mit.edu/docs/iii/tables/

# Requirements / Pre-requisite

Install Python & required libraries (Code file includes a subsection for installation using PIP)

# How to Execute


#### Step-1 Clone the git repo to your machine

#### Step-2: Download the data files
Download the data files from below google drive location to your machine where you plan to execute the code:

https://drive.google.com/drive/folders/1Lsjg3Kl93L0pn5EYYsilsuX6Yl4d4Zy4?usp=sharing

Preferably place the downloaded files under: /dlh-project-sp23/StudentCode/InputData/

#### Step-3: Set the variables
Before you execute the code, you need to set couple of variables in the code file as follows:

1. data_dir: Once you download the data files as instructed above, place them in desired folder on your machine and provide a directory path here. (If used preferred path then no need for any changes)
2. labevents_records: If you are executiong Version 2 above (rnn_gru_relu.py) then you need to set this according to the computation power you have on your machine. Default is 500k (500000) and Max is 27.8M (27854060). If you are executing on personal computer then I would recommend keep it as default.
3. inputdf_records: Set this according to the computation power of your machine. Default is 15k and Max is 1.4M for Verison 1 and 200k for Version 2. If you are executing on personal computer then I would recommend keep it as default.
4. max_epoch: Default is 3. You can set this as per your preference.

#### Step-4: Executing the code:

1. Python rnn_gru_simoid.py
2. Python rnn_gru_relu.py

