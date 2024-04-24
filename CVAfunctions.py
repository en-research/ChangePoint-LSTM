
"""
Portions of code take insipiration and are modified from the following sources (originally in Matlab):
1. https://ww2.mathworks.cn/matlabcentral/fileexchange/50938-a-benchmark-case-for-statistical-process-monitoring-cranfield-multiphase-flow-facility?s_tid=srchtitle

"""

# Import libraries
import numpy as np
import scipy as sp

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from shapely.geometry import LineString   # For finding intersection of threshold and T2/Q statistic


################################### DATA PREPROCESSING ########################################

def select_engines(dataset, min_timespan):
    # Selects Engines with minimum timespan (e.g.>=200) 
    # Returns dataframe with data of selected engines and also array of the engine numbers
    data_min_timespan = dataset.loc[dataset['Max_cycle'] >= min_timespan]
    engine_list = data_min_timespan['Unit_No'].unique() # List of engine numbers meeting min timespan condtion
    
    # Find corresponding max cycles for the selected engines
    df = data_min_timespan.groupby('Unit_No')
    engine_maxcycle_list = df['Time_Cycles'].max() 
    
    return data_min_timespan, engine_list, engine_maxcycle_list 
    

def create_CVATrainTestdata(data_min_timespan, unit_no, end_time_train, end_time_val):
    # Returns X1 (training), X2 (validation) and Xt (test data)
    # X1 and X2 forms the normal operation data
    
    # For each engine, drop all columns except for helpful sensors
    data = data_min_timespan.loc[data_min_timespan['Unit_No'] == unit_no].reset_index(drop=True)
    data.drop(['Max_cycle', 'Time_Cycles', 'Unit_No', 'RUL'], axis= 1, inplace = True)
    
    # Create training data for normal operating condition (first n cycles)
    data_train = data.iloc[:end_time_train]
    X1 = np.array(data_train)
    
    # Create validation data (next k cycles)
    data_val = data.iloc[end_time_train:end_time_val].reset_index(drop=True)
    X2 = np.array(data_val)
    
    # Create testing data (remaining cycles)
    data_test = data.iloc[end_time_val:].reset_index(drop=True)
    Xt = np.array(data_test)

    return X1, X2, Xt



############################# HANKEL FUNCTION PAST AND FUTURE ##################################

def hankelpf(y,p,f=0):
    # number of observations
    N, ny = np.shape(y)
    
    # Ip = flipud(hankel(1:p,p:N-f)) 
    # Py index starts from 0 and end index is exclusive. Therefore slice index adjusted so that Ip picks out correct value in Xp[Ip,:] 
    Hp = sp.linalg.hankel(np.arange(0,p) , np.arange(p-1, N - f))  # Create Hankel matrix
    Ip = np.flipud(Hp)          # indices of past observations
    
    A = y[Ip,:]  # Select out matrix A from y using Ip indices; Note: A is just an arbitary variable name
    
    # Equivalent to Yp = np.reshape(c1,(ny*p, -1))
    # Important Note: this Yp1 is hardcoded concatanation for the case where Yp1 has 2 rows (p=f=2); For any other values than 2, a more general formula representation is needed (to check and revise later e.g., Yp1 =A.reshape([ny*p, -1]).
    Yp1 = np.concatenate((A[0,:,:].T, A[1,:,:].T), axis =0 ) 
    
    if f > 0:
        #  If = hankel(p+1:p+f,p+f:N);  
        If = sp.linalg.hankel(np.arange(p+1-1,p+f) , np.arange(p+f-1,N)) # indices of future observations
        
        cf = y[If,:]
        
        # Equivalent to Yf = reshape(y(If,:)',ny*f,[]);
        Yf1 = np.concatenate((cf[0,:,:].T, cf[1,:,:].T), axis =0 )
        
        return Yp1, Yf1

    else:
        return Yp1
    


############################# KDE FUNCTION ##################################

def gkde_find_threshold(x, alpha = 0.99): # x is either T or Q # Default alpha is 0.99
    
    # Features of given data
    x = x.flatten(order ='F')    # where x can be T or Q statistic #Creates a linear column vector like matlab code x= x(:) but it does the job for now because x is 1 x 74 array
    
    # Above line gives only 1 d array (no. of rows,); Need 2d array (o. of rows,1) for kde.fit()
    x = x.reshape(x.shape[0],1) # dimension: (no. of rows, 1)
    
    # No. of elements in x; Value will be used to create N uniformly spaced values for p_x
    N = np.size(x)
    
    n = np.size(x) # To be used in bandwidth h calculation following matlab code 
    
    # Bandwidth h
    h = np.median(abs(x - np.median(x)))*1.5704/(n**0.2)  # Copied from Matlab code, some sort of Mean Absolute deviation
    dx= h * 3
    p_x = np.linspace(min(x)-dx, max(x)+dx, N)  # Transposed to make it same as matlab output 1 x100
    kde = KernelDensity(kernel='gaussian', bandwidth= h)
    kde.fit(x)
    log_dens = kde.score_samples(p_x)  # Gives log of pdf value
    pdf = np.exp(log_dens)             # To get actual pdf values
    
    cdf = np.cumsum((np.append(0, np.diff(p_x, n=1, axis = 0))) * pdf) #delta x * pdf = area under curve for that section; Add up the sections cumulatively to get cdf
    
    index = np.where(cdf <= alpha) # Find index where cdf is  <0.99
    values = p_x[index]        # Find all the values of x in p_x that match index above
    control_limit = max(values)    # Find max of all the values, that is the threshold/ Control limit of T2
    
    # Reshape from a single value to an array of that same value so the threshold can be plotted as a horizontal line later
    # max_x = max_x * np.ones((N,1))
    return control_limit



####################### FIND CHANGEPOINT (AFTER TRANSITION PERIOD) FOR TRAIN ENGINES ################################## 

# Function returns the last intersection point (Time cycle) of T2/Q with the control limit
# Input:
# data: T2 or Q from test data (where fault begins to occur after some transient fluctuations about threshold)
# control limit: threshold T2 or Q

def change_point(data, control_limit, train_samples = 60, val_samples = 20):
    data = data.T    # Convert data from (1,n) to (n,1) so that it matches with first index of dimensions of sample (n,). This is a requirement for the LineString column stack command
    N = np.size(data)
    sample = np.arange(1, N + 1)
    c_limit_array = control_limit * np.ones((N,1))

    first_line = LineString(np.column_stack((sample, data)))
    second_line = LineString(np.column_stack((sample, c_limit_array)))
    intersection = first_line.intersection(second_line)
    
    x_coord_intersect, _ = LineString(intersection).xy

    return round(max(x_coord_intersect)) + train_samples + val_samples

####################### FIND CHANGEPOINT (BEFORE TRANSITION PERIOD) FOR TRAIN ENGINES ################################## 

# Find the changepoint before transition period
# Function returns the first intersection point (Time cycle) of T2/Q with the control limit
# Input:
# data: T2 or Q from test data (where fault begins to occur after some transient fluctuations about threshold)
# control limit: threshold T2 or Q

def change_point_trans(data, control_limit, train_samples = 60, val_samples = 20):
    data = data.T    # Convert data from (1,n) to (n,1) so that it matches with first index of dimensions of sample (n,). This is a requirement for the LineString column stack command
    N = np.size(data)
    sample = np.arange(1, N + 1)
    c_limit_array = control_limit * np.ones((N,1))

    first_line = LineString(np.column_stack((sample, data)))
    second_line = LineString(np.column_stack((sample, c_limit_array)))
    intersection = first_line.intersection(second_line)
    
    x_coord_intersect, _ = LineString(intersection).xy

    return round(min(x_coord_intersect)) + train_samples + val_samples



############################ FIND TRANSITION PERIOD ################################## 
# Function returns the first and last intersection point (Time cycle) of T2/Q with the control limit
# Input:
# data: T2 or Q from test data (where fault begins to occur after some transient fluctuations about threshold)
# control limit: threshold T2 or Q

def transition_period(data, control_limit, train_samples = 60, val_samples = 20):
    data = data.T    # Convert data from (1,n) to (n,1) so that it matches with first index of dimensions of sample (n,). This is a requirement for the LineString column stack command
    N = np.size(data)
    sample = np.arange(1, N + 1)
    c_limit_array = control_limit * np.ones((N,1))

    first_line = LineString(np.column_stack((sample, data)))
    second_line = LineString(np.column_stack((sample, c_limit_array)))
    intersection = first_line.intersection(second_line)
    
    x_coord_intersect, _ = LineString(intersection).xy

    return round(min(x_coord_intersect)) + train_samples + val_samples, round(max(x_coord_intersect)) + train_samples + val_samples



############################ PLOT T2 AND Q ##################################    

def statistic_plot(data_T, data_Q, control_limit_T, control_limit_Q, plot_title, xlabel = 'Time cycle', ylabel_T ='T2', ylabel_Q ='Q'):
    
    
    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex = False)
    
    # Plot for T2
    N = np.size(data_T)
    data = data_T.T   # In order for x and y variables to have same first dimension
    sample_no = np.arange(1, N + 1) # This will be the x-axis
    
    ax1.plot(sample_no, data, color="b")
    ax1.hlines(control_limit_T, xmin = min(sample_no), xmax = max(sample_no), colors="r", linestyles='--')
    
    #ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_T)
    ax1.set_yscale("log")

    # Plot for Q
    N = np.size(data_Q)
    data = data_Q.T   # In order for x and y variables to have same first dimension
    sample_no = np.arange(1, N + 1) # This will be the x-axis
    
    ax2.plot(sample_no, data, color="b")
    ax2.hlines(control_limit_Q, xmin = min(sample_no), xmax = max(sample_no), colors="r", linestyles='--')
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel_Q)
    ax2.set_yscale("log")      
   
    
    fig.suptitle(plot_title)
    plt.tight_layout()
    #plt.show()
    exec(f"fig.savefig('{plot_title}.png')")



####################### CVA TUTOR FUNCTION FOR ENGINES ###############################    
        

def CVAtutor(X1,X2,Xt, a=0.99,n=15,p=2,f=0, m=14, plt_title ='Training'):
    # Inputs:
    # X1:   first training data set . Each row is an observation, each column is a variable
    # X2:   second training data set . Each row is an observation, each column is a variable
    # Xt:   test data. Each row is an observation, each column is a variable
    # a:    confidence level, 99% by default
    # n:    retained state dimension (r in our paper)
    # p:    length of past observation
    # f:    length of future observation
    # m:    No. of feature variables

    # Outputs:
    # Plots and saves T2 and Q plots for training/validation/ test data
    # Also returns Ta, Qa, changepoint_T, changepoint_Q
    

    ############## CREATE PAST AND FUTURE MATRIX Yp AND Yf ##################
    Yp1, Yf1 = hankelpf(X1,p=2,f=2) 
    Yp2, Yf2 = hankelpf(X2,p=2,f=2)
    
    Yp = np.concatenate((Yp1,Yp2), axis =1)
    Yf = np.concatenate((Yf1,Yf2), axis =1)
    
    ############## NORMALISATION OF Yp AND Yf ##################
    
    # Normalisation of past and future matrix
    pn = Yp.shape[1] # Extract out number of columns
    
    # For numpy std, ddof =1 to be same as Matlab std 
    fmean = Yf.mean(axis=1).reshape(Yf.mean(axis=1).shape[0],1) # from (r,) to (r,1) where r is no. of rows
    fstd = Yf.std(axis=1,ddof = 1).reshape(Yf.std(axis=1).shape[0],1)
    pmean = Yp.mean(axis=1).reshape(Yp.mean(axis=1).shape[0],1)
    pstd = Yp.std(axis=1, ddof = 1).reshape(Yp.std(axis=1).shape[0],1)
    
    # Normalisation of past and future matrix
    # Reshape fmean, fstd, pmean, pstd to (28,pn) by element multiplying with np.ones
    Yp = (Yp - pmean * np.ones((m*p,pn)))/(pstd * np.ones((m*p,pn))) 
    Yf = (Yf - fmean * np.ones((m*p,pn)))/(fstd * np.ones((m*p,pn)))
    
    ############## CHOLESKY AND HANKEL MATRIX ##################
    
    # Obtain Cholesky matrices and Hankel matrix
    Rp = sp.linalg.cholesky(np.dot(Yp,Yp.T)/(pn-1)) # Past Cholesky matrix
    Rf = sp.linalg.cholesky(np.dot(Yf,Yf.T)/pn)     # Future Cholesky matrix
    Hfp = np.dot(Yf,Yp.T)/pn                        # Cross-covariance matrix
    numerator_H = np.dot(sp.linalg.inv(Rf.T),Hfp)   # Equivalent to matlab (Rf.T\Hfp)
                                                    # It seems that / in python does elementwise division
    
    H = np.dot(numerator_H,sp.linalg.inv(Rp))       # Hankel matrix # Equivalent to matlab H = (Rf.T\Hfp)/Rp                          
    
    
    ################# SVD and CVA ##########################
    
    _ , S, V = sp.linalg.svd(H)         # SVD
    V = V.T                             # To get transposed V as matlab
    S = S.reshape(S.shape[0],1)         # No need for S = diag(S) as in Matlab as Python svd only return  (28,) for S and not (28,28)
                                        # Reshape from (28,) to (28,1)
    m1 = np.size(S)                     # Equivalent to m = numel(S)? Finding no. of elements in array; Cant name this variable as m as m is taken up for another input variable in CVAtutor
    
    
    
    V1 = V[:,0:n]                         # Reduced V matrix # Equivalent to matlab V1 = V(:,1:n)
    J = np.dot(V1.T, sp.linalg.inv(Rp.T)) # Transformation matrix of state variables
                                          # Equivalent to J = V1'/Rp'
    
    numerator_L = np.eye(m1) - np.dot(V1,V1.T)                       
    L = np.dot(numerator_L, sp.linalg.inv(Rp.T))  # Transformation matrix of residuals
                                                  # Equivalent to L = (eye(m)-V1*V1')/Rp'
    z = np.dot(J,Yp)                              # States of training data
    e = np.dot(L,Yp)                              # Residuals of training data    
    
    
    T= np.sum(z*z, axis =0)             #  T^2 of training data 
                                        # Note: In Matlab, z.*z  refers to elementwise multiplication 
    T = T.reshape(1, T.shape[0])        # Reshape from (74,) to same shape as Matlab (1,74) 
    
    Q = np.sum(e*e, axis =0)            # Q statistic of training data
                                        # Note: In Matlab, e.*e  refers to elementwise multiplication 
    Q = Q.reshape(1, Q.shape[0])        # Reshape from (74,) to same shape as Matlab (1,74)
    
    
    ################# COMPUTE KDE BASED CONTROL LIMITS ##########################
    
    Ta = gkde_find_threshold(x = T, alpha = a)  # Threshold/control limit for 99% significance level for T2 distribution
    Qa = gkde_find_threshold(x = Q, alpha = a)  # Threshold/control limit for 99% significance level for Q distribution
    
    
    ################# MONITORING (TEST DATA) ##########################
    
    Ypm = hankelpf(Xt,p)   # hankelp(y,p) 
    
    # Normalisation of past test observation matrix
    pn_m = Ypm.shape[1] # Extract out number of columns
    
    # Reshape pm_mean, pm_std to (28, pn_m) by element multiplying with np.ones
    Ypmn = (Ypm - pmean * np.ones((m*p,pn_m)))/(pstd * np.ones((m*p,pn_m)))
    
    
    # Compute T^2 and Q indicators for monitoring
    
    zk = np.dot(J,Ypmn)                   # States of test data
    ek = np.dot(L,Ypmn)                   # Residuals of test data    
    
    
    T2mon = np.sum(zk*zk, axis =0)           #  T^2 of test data 
                                             # Note: In Matlab, z.*z  refers to elementwise multiplication 
    T2mon = T2mon.reshape(1, T2mon.shape[0]) # Reshape from (pn_m,) to same shape as Matlab (1,pn_m) 
    
    Qmon = np.sum(ek*ek, axis =0)            # Q statistic of test data
                                             # Note: In Matlab, e.*e  refers to elementwise multiplication 
    Qmon = Qmon.reshape(1, Qmon.shape[0])    # Reshape from (pn_m,) to same shape as Matlab (1,pn_m)

    
    ################# PLOT AND SAVE T2 AND SPE FOR MONITORING DATA ##########################
    # Uncomment statistic_plot for visualing every engine monitoring data
    #statistic_plot(data_T =T2mon, data_Q = Qmon, control_limit_T = Ta, control_limit_Q = Qa, plot_title = plt_title) #plt_title comes from input to CVA Tutor function
    

   #return Ta, Qa, changepoint_T, changepoint_Q
    return Ta, Qa, T2mon, Qmon


        

############################ CLIP RUL ################################## 

def clip_RUL(df_left, df_right, eng_list, same_upperRUL = False):
    # Function adds changepoint, upper RUL (value to clip RUL by) and clips RUL
    # If same_upperRUL is True 
    
    df_merge = pd.merge(left= df_left, right = df_right[['Unit_No','Early_Changepoint']], on = 'Unit_No', how = 'outer')

    # For engines (timespan<200 cycles) where changepoint was not calculated, changepoint of 130 assigned
    #X_train_FD1_merge['Early_Changepoint'].fillna(value = 130, inplace = True)
    
    # Calculate upper limit of RUL for engines with changepoints calculated
    df_merge['Upper_RUL'] = df_merge['Max_cycle'] - df_merge['Early_Changepoint']
     
    if same_upperRUL:
        for i in df_merge['Unit_No'].unique():
            
            df_merge.loc[df_merge['Unit_No']==i, ['RUL']] = df_merge.loc[df_merge['Unit_No']==i, ['RUL']].clip(upper=130)
            
            # Assign upper 'Upper_RUL' column with 130
            df_merge.loc[df_merge['Unit_No']==i, ['Upper_RUL']] = 130
                
            # Add back calculated changepoints 
            # Changepoint = Max cycle - Upper_RUL (130 in this case)
            df_merge.loc[df_merge['Unit_No']==i, ['Early_Changepoint']] = df_merge['Max_cycle'] - df_merge['Upper_RUL']

        
    else:
        # Clip RUL based on changepoint for 2 cases
        
        # Case 1: For engines without changepoints calculated, assume RUL starts degrading after last 130 cycles similar to benchmark papers.
        for i in df_merge['Unit_No'].unique():
            if not i in eng_list:
                df_merge.loc[df_merge['Unit_No']==i, ['RUL']] = df_merge.loc[df_merge['Unit_No']==i, ['RUL']].clip(upper=130)
                
                # Assign upper 'Upper_RUL' column with 130
                df_merge.loc[df_merge['Unit_No']==i, ['Upper_RUL']] = 130
                
                # Add back calculated changepoints 
                # Changepoint = Max cycle - Upper_RUL (130 in this case)
                df_merge.loc[df_merge['Unit_No']==i, ['Early_Changepoint']] = df_merge['Max_cycle'] - df_merge['Upper_RUL']
        
        # Case 2: For engines changepoints calculated, assume RUL starts degrading after change point
        for i in eng_list:
            df_merge.loc[df_merge['Unit_No']==i, ['RUL']] = df_merge.loc[df_merge['Unit_No']==i, ['RUL']].clip(upper= df_merge['Upper_RUL'], axis =0)
        
        # Add backcalculated changepoints for engines without CVA-based changepoints
        # Changepoint = Max cycle - 130
        
        #for i in df_merge['Unit_No'].unique():
           # if not i in eng_list:
              #  df_merge.loc[df_merge['Unit_No']==i, ['Early_Changepoint']] = df_merge.loc[df_merge['Unit_No']==i, ['Max_cycle']] - 130
        
    return df_merge


