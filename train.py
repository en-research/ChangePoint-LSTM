
"""
Portions of code take insipiration and are modified from the following sources:
1. https://towardsdatascience.com/lstm-for-predictive-maintenance-of-turbofan-engines-f8c7791353f3
2. https://github.com/schwxd/LSTM-Keras-CMAPSS

"""
import argparse
import numpy as np
import scipy as sp
import random
import time 
import pandas as pd
import matplotlib.pyplot as plt
import CVAfunctions 
from CVAfunctions import select_engines, create_CVATrainTestdata, CVAtutor, change_point,change_point_trans, clip_RUL

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import h5py

#%%
# For reproducible results
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Note: more hyperparameters can be added to the argument parser (currently only r and min_lifespan are included)
parser = argparse.ArgumentParser()
parser.add_argument('--r', type=int, default=15, help="number of system canonical variates")
parser.add_argument('--min_lifespan', type=int, default=200, help="min lifespan of train engines")
args = parser.parse_args()


# Variables to adjust

FD_data = 1 #  Enter 1,2,3 or 4 for FD001-FD004 dataset
boolean_sameUpper = False #If  true, cap RUL of all engines to 130 (i.e. no individual changepoint detected)
boolean_transition = False # If true, changepoint is defined before transition period, else after transition period


r = args.r #Number of system canonical variates
min_lifespan = args.min_lifespan  # minimum lifespan of engines used for change point detection training
seq_length = 50
bat_size = 64
epoch = 30

#opt = 'Adam' # optimiser for LSTM; Alternative is RMSprop
#lr = 0.001    # learning rate

############################ DATA PREPROCESSING  ############################


# Read train and test data from text file
direct_path = './CMAPSSData/' 
train_file = f'train_FD00{FD_data}.txt'
test_file = f'test_FD00{FD_data}.txt'
RUL_file = f'RUL_FD00{FD_data}.txt'

# Directory to save model
if boolean_transition: # change point defined before transition period
    save_dir = f'.saved_models_FD{FD_data}_trans/'

else: # change point defined after transition period
    save_dir = f'.saved_models_FD{FD_data}/' 
    
# Create column names
index_names = ['Unit_No', 'Time_Cycles']
setting_names = [f'Setting_{i+1}' for i in range(0,3)] 
sensor_names = [f'Sensor_{i+1}' for i in range(0,21)]

col_names = index_names + setting_names + sensor_names


X_train_FD1 = pd.read_csv(direct_path + train_file, sep = '\s+', header= None, names= col_names)
X_test_FD1 = pd.read_csv(direct_path + test_file, sep = '\s+', header= None, names= col_names)

y_RUL_FD1 = pd.read_csv(direct_path + RUL_file, sep = '\s+', header= None, names= ['RUL'])

# X_train_FD1.head


############################ ADD RUL COLUMN  ############################

# Define RUL as Max cycle - Current time cycle

def add_RUL(df):
    grouped_by_unit_no = df.groupby('Unit_No')
    max_cycle = grouped_by_unit_no['Time_Cycles'].max() # A Series where Unit_No is the index value and Time_cycles is a Column
    
    # Rename columnS
    max_cycle = pd.DataFrame(max_cycle) # Change series to dataframe for rename function to work as Series are not considered to have columns
    max_cycle.rename(columns = {'Time_Cycles': 'Max_cycle'},inplace = True)
    
    # Merge max_cycle to original dataframe, df
    df_merge = pd.merge(left= df, right = max_cycle, on = 'Unit_No', how = 'outer')
    
    # Create a new column,'RUL'
    df_merge['RUL'] = df_merge['Max_cycle'] - df_merge['Time_Cycles']
    
    # Drop Max_cycle column as it is no longer needed
    #df_merge.drop('Max_cycle', axis=1, inplace = True) # Dont drop max cycle yet as we need to select engines with more than 200 cycles
    
    return df_merge

# Add the RUL column
X_train_FD1= add_RUL(X_train_FD1)

# Find ave lifespan of engines
#grouped_by_unit_no = X_train_FD1.groupby('Unit_No')['Time_Cycles'].max()

Ave_lifespan = X_train_FD1.groupby('Unit_No')['Max_cycle'].max().mean()


############################ DROP UNHELPFUL SENSORS  ############################


# Inspect sensor values, specifically if there are any with near 0 standard deviation 
# sensors readings which do not change over time hold no useful information
X_train_FD1[sensor_names].describe().transpose()

# Based on our Exploratory Data Analysis (for FD01 and FD03) we can determine:
# Sensors 1, 5, 16, 18 and 19 are constant and hold no information related to RUL as the sensor values remain constant throughout time. 
# Sensors 6, 10 are irregular

# For FD02 and FD04, the constant/irregular sensors are 13,10,16,18,19

# Drop irrelevant variables
if FD_data==1 or FD_data==3:
    unhelpful_sensors = [f'Sensor_{i}' for i in [1,5,6,10,16,18,19]]
else: #FD02 and FD04
    unhelpful_sensors = [f'Sensor_{i}' for i in [13,10,16,18,19]]

helpful_sensors = [x for x in sensor_names if x not in unhelpful_sensors]
drop_labels = setting_names + unhelpful_sensors
X_train_FD1.drop(drop_labels, axis=1, inplace = True)


############################ PREPROCESS TEST DATA  ############################


# Drop the same unhelpful labels 
test_data = X_test_FD1.drop(drop_labels, axis=1) # Recall: drop_labels = setting_names + unhelpful_sensors



#################### SELECT ENGINES WITH AT LEAST 200 CYCLES ##################

eng_more200, engine200, engine200_maxcyc = select_engines(dataset = X_train_FD1, min_timespan = min_lifespan)


#################### CREATE DATASET AND T2 , Q PLOTS ##################

Ta_AllEngines = []
Qa_AllEngines = []
changepoint_T_AllEngines = []
changepoint_Q_AllEngines = []

# Use correct number of sensors depending on FD01, 02, 03 or 04
if FD_data==1 or FD_data==3:
    num_sensors = 14
else: #FD02 and FD04
    num_sensors = 16

# Determine changepoints
for i in engine200:
    X1, X2, Xt = create_CVATrainTestdata(data_min_timespan = eng_more200, unit_no = i, end_time_train = 60, end_time_val= 80)
    
    # Training cycles Plot
    #CVAtutor(X1 = X1, X2 = X2, Xt = X1 , a=0.99,n=15,p=2,f=0, m= num_sensors, plt_title = f'Training (Engine {i})')
    
    # Validation cycles Plot
    #CVAtutor(X1 = X1, X2 = X2, Xt = X2, a=0.99,n=15,p=2,f=0, m= num_sensors, plt_title = f'Validation (Engine {i})')
    
    # Test cycles Plot and changepoint calculation
    Ta, Qa, T2mon, Qmon = CVAtutor(X1 = X1, X2 = X2, Xt = Xt, a=0.99,n= r,p=2,f=0, m= num_sensors, plt_title = f'Test (Engine {i})')
    
    # Collate Ta and Qa of all engines in a list
    Ta_AllEngines.append(np.ndarray.item(Ta)) 
    Qa_AllEngines.append(np.ndarray.item(Qa))
    
    if boolean_transition:
        # Changepoint calculated (before transition state)
        changepoint_T = change_point_trans(data = T2mon, control_limit = Ta, train_samples = 60, val_samples = 20)    # Change point based on T2 plot and intersection with threshold Ta
        changepoint_Q = change_point_trans(data = Qmon, control_limit = Qa, train_samples = 60, val_samples = 20)     # Change point based on Q plot and intersection with threshold Qa
    
    else:
        #Changepoint calculated (after transition state)
        changepoint_T = change_point(data = T2mon, control_limit = Ta, train_samples = 60, val_samples = 20)    # Change point based on T2 plot and intersection with threshold Ta
        changepoint_Q = change_point(data = Qmon, control_limit = Qa, train_samples = 60, val_samples = 20)     # Change point based on Q plot and intersection with threshold Qa


    # Collate changepoints of all engines in a list
    changepoint_T_AllEngines.append(changepoint_T)
    changepoint_Q_AllEngines.append(changepoint_Q)


#################### CREATE DATAFRAME OF CHANGEPOINTS ##################


d = {'Unit_No': engine200, 'Ta': Ta_AllEngines, 'Qa': Qa_AllEngines, 'Changepoint_T': changepoint_T_AllEngines, 'Changepoint_Q': changepoint_Q_AllEngines, 'Max_cycle': engine200_maxcyc}
df_n15 = pd.DataFrame(data=d).reset_index(drop = True)

# Pick out the changepoint that occurs earlier (because early warning preferred)
df_n15['Early_Changepoint'] = df_n15[['Changepoint_T','Changepoint_Q']].min(axis=1)



#################### DATA PREP FOR RUL ESTIMATION ##################

# Add CVA-determined changepoint, upper RUL (value to clip RUL by) and clips RUL
# For the short-lifespan engines where CVA-determined changepoints were not calculated, changepoint is back calculated from upper RUL limit of 130

X_train_FD1_merge = clip_RUL(df_left = X_train_FD1, df_right= df_n15, eng_list = engine200, same_upperRUL= boolean_sameUpper)


#################### CREATE TRAIN AND TEST DATASETS ##################
# Standardise all sensor columns in train data with mean and standard deviation before changepoint


train_data = X_train_FD1_merge.copy()  # Make a copy of the df so you can compare the normalised columns with the original columns
#train_data[helpful_sensors] = std.fit_transform(train_data[helpful_sensors])

# Goal is to find mean and std of data of all engines before changepoint
# Collate normal range (before changepoint) data of all engines into a dataframe
train_beforeCP = []
for i in train_data['Unit_No'].unique():
    changepoint = int(train_data.loc[train_data['Unit_No']==i, ['Early_Changepoint']].iloc[0]) #For each engine, all numbers in the column 'Early_changepoint' are the same, so  just extract the value from first one. Just need a single value; cast the value as integer, not series  
    
    train_beforeCP.append(train_data.loc[train_data['Unit_No']== i,:].iloc[:changepoint] )

train_beforeCP_array = np.concatenate(train_beforeCP)
train_beforeCP_df = pd.DataFrame(train_beforeCP_array, columns = train_data.columns)

#################### STANDARDISE TRAIN AND TEST DATA ##################


# Create standard scalar object and fit with train data in normal range (before changepoint)
scalar = StandardScaler()
scalar.fit(train_beforeCP_df[helpful_sensors])

mean_train_beforeCP = scalar.mean_  # mean of each of the 14 sensors in across all engines
std_dev_train_beforeCP = np.sqrt(scalar.var_)

# Transform train and test data
train_data[helpful_sensors] = scalar.transform(train_data[helpful_sensors])

test_data[helpful_sensors] = scalar.transform(test_data[helpful_sensors])

#train_data.drop(['Time_Cycles', 'Max_cycle', 'Early_Changepoint', 'Upper_RUL'], axis = 1, inplace=True)

# Create X_train (features) and y_train (label). Note that 'Unit_No' is still in the dataset for creating the LSTM input data below
X_train = train_data[['Unit_No'] + helpful_sensors].copy()
y_train = train_data[['Unit_No','RUL']].copy()





#################### FUNCTIONS TO PREPARE INPUTS TO LSTM MODEL ##################

# Create sequence of training data for features
def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]

    for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
        yield data[start:stop, :]  


def gen_train_data_wrapper(df, sequence_length, columns, unit_no = np.array([]), batch_size=32):
    
    if unit_no.size <= 0: # The default input to gen_train_data_wrapper function
        unit_nos = df['Unit_No'].unique()
        #print(unit_nos)
    
    data_gen = [list(gen_train_data(df[df['Unit_No']== i], sequence_length, columns)) 
                for i in unit_nos if len(df[df['Unit_No']  == i]) > sequence_length]
    

    data_array = np.concatenate(list(data_gen)).astype(np.float32)

    return data_array

    
# Create sequence of training data for label (i.e. RUL)
def gen_labels(df, sequence_length, columns = ['RUL']):
    data = df[columns].values
    num_elements = data.shape[0]

    return data[sequence_length-1:num_elements, :]  


def gen_label_wrapper(df, sequence_length, columns, unit_no=np.array([]), batch_size = 32):
    if unit_no.size <= 0:
        unit_nos = df['Unit_No'].unique()
        
    label_gen = [gen_labels(df[df['Unit_No']== i], sequence_length) 
                for i in unit_nos]
    label_array = np.concatenate(label_gen).astype(np.float32)

    return label_array 


X  = gen_train_data_wrapper(df = X_train, sequence_length= seq_length, columns= helpful_sensors)

y = gen_label_wrapper(df = y_train, sequence_length= seq_length, columns= ['RUL'])


# Create sequence of test data by padding 
def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value) # pad
        idx = data.shape[0] - df.shape[0]
        data[idx:,:] = df[columns].values  # fill with available data
    else:
        data= df[columns].values
        
    # specifically yield the last possible sequence (e.g. if sequence length is 30, then last 30 cycles because we only predict based on last cycle)
    stop = num_elements = data.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data[start:stop, :]

def gen_test_data_wrapper(df, sequence_length, columns, mask_value):
    
    test_gen = (list(gen_test_data(df[df['Unit_No']== i], sequence_length, columns, mask_value))
           for i in df['Unit_No'].unique())
    test_array = np.concatenate(list(test_gen)).astype(np.float32)
    
    return test_array

X_test = gen_test_data_wrapper(test_data, sequence_length = seq_length, columns = helpful_sensors, mask_value =-99.)

y_test = np.array(y_RUL_FD1).astype(np.float32)


#################### LSTM MODEL ##################

print(X.shape)
print(y.shape)
print(X_test.shape)
print(y_test.shape)


timesteps = X.shape[1]
input_dim = X.shape[2]

#3 layer model
model = tf.keras.models.Sequential()
model.add(Masking(mask_value=-99., input_shape=(timesteps, input_dim)))
model.add(tf.keras.layers.LSTM(input_shape=(timesteps, input_dim), units=256, return_sequences=True, activation = 'sigmoid' ))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(input_shape=(timesteps, input_dim), units=128, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.LSTM(units=32, return_sequences=False))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

#2 layer model
# model = tf.keras.models.Sequential()
# model.add(Masking(mask_value=-99., input_shape=(timesteps, input_dim)))
# model.add(tf.keras.layers.LSTM(input_shape=(timesteps, input_dim), units=256, return_sequences=True, activation = 'relu' ))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.LSTM(input_shape=(timesteps, input_dim), units=100, return_sequences=False))
# #model.add(tf.keras.layers.Dropout(0.2))
# #model.add(tf.keras.layers.LSTM(units=32, return_sequences=False))
# model.add(tf.keras.layers.Dense(units=1, activation='linear'))

#1 layer model
# model = tf.keras.models.Sequential()
# model.add(Masking(mask_value=-99., input_shape=(timesteps, input_dim)))
# model.add(tf.keras.layers.LSTM(input_shape=(timesteps, input_dim), units=256, return_sequences=False, activation = 'sigmoid' ))
# # model.add(tf.keras.layers.Dropout(0.2))
# # model.add(tf.keras.layers.LSTM(input_shape=(timesteps, input_dim), units=100, return_sequences=False))
# #model.add(tf.keras.layers.Dropout(0.2))
# #model.add(tf.keras.layers.LSTM(units=32, return_sequences=False))
# model.add(tf.keras.layers.Dense(units=1, activation='linear'))


model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
learning_rate = tf.keras.backend.get_value(model.optimizer.lr)
print(model.summary())
print(learning_rate)

# Patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

checkpoint = ModelCheckpoint(save_dir+'best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
callbacks_list = [checkpoint, es]
    

# Start timing model
t = time.time()

# Train model
history = model.fit(X, y, batch_size = bat_size, epochs = epoch,  validation_data = (X_test, y_test), callbacks=callbacks_list)

# Stop timing model
elapsed_time = round((time.time() - t)/60)

# Load best model to evaluate the performance of the model
model.load_weights(save_dir + "best_model.h5")


# Predict RUL (RUL is clipped to upper limit 130) following existing works such as https://www.sciencedirect.com/science/article/pii/S0951832021004439?via%3Dihub (for discussion refer to change point paper)
y_pred = pd.DataFrame(model.predict(X_test)).clip(upper=130)
y_true = y_RUL_FD1.clip(upper=130)


mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Scoring function
h = np.array(y_pred) - np.array(y_true)

scoring = 0
for i in range(len(h)):
  if h[i] <0:
    score = np.exp((-1/13)* h[i]) - 1
  if h[i] >=0:
    score = np.exp((1/10)* h[i]) - 1
  scoring += score
  #print(h[i])

# Print results
print(f'\
      boolean_sameUpper: {boolean_sameUpper}\
      sequence_length: {seq_length}\
      batch size: {bat_size}\
      epochs: {epoch}\
      Training time: {elapsed_time} mins'
      )

print(f' FD:{FD_data} Loss(MSE): {mse}  RMSE: {rmse}  Scoring: {scoring}')


