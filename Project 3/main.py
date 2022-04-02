# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 00:01:12 2021

@author: wayne
"""

import pandas as pd
import numpy as np
import math
from scipy.fftpack import fft, rfft
from sklearn.cluster import KMeans, DBSCAN
# from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn import metrics

#load Insulin and CGM Data
insulin_df = pd.read_csv('InsulinData.csv', parse_dates=[['Date','Time']],low_memory=False).iloc[::-1]
cgm_df = pd.read_csv('CGMData.csv', parse_dates=[['Date','Time']],low_memory=False).iloc[::-1]

insulin_df = insulin_df[['Date_Time','BWZ Carb Input (grams)']]
insulin_Meal_df= insulin_df[insulin_df['BWZ Carb Input (grams)'].notnull() & insulin_df['BWZ Carb Input (grams)'] != 0]

insulin_Meal_df.loc[:, 'Time_dif1'] = insulin_Meal_df['Date_Time'].diff(periods=-1)/np.timedelta64(1,'h')
insulin_Meal_df.loc[:,'Time_dif2'] = insulin_Meal_df['Date_Time'].diff(periods=1)/np.timedelta64(1,'h')

filt=((insulin_Meal_df['Time_dif1']<-2) | (insulin_Meal_df['Time_dif2'].iloc[-1:]>2))
insulin_Meal_df = insulin_Meal_df.loc[filt]

binSize = 20
MaxVals = insulin_Meal_df['BWZ Carb Input (grams)'].max()
MinVals = insulin_Meal_df['BWZ Carb Input (grams)'].min()

# nBins = math.floor((MaxVals-MinVals)/20)
nBins = math.ceil((MaxVals-MinVals)/20)

# insulin_Meal_df['Ground Truth Label'] = ((insulin_Meal_df['BWZ Carb Input (grams)']-MinVals)/20).apply(np.floor)
insulin_Meal_df['Ground Truth Label'] = ((insulin_Meal_df['BWZ Carb Input (grams)']-MinVals)/20).apply(np.ceil)

#CGM meal datetime
CGM_Meal_Date_Time = []

for datetime in insulin_Meal_df['Date_Time']:
    l = []
    CGM_Meal_Datetime = cgm_df.loc[cgm_df['Date_Time']>=datetime]['Date_Time'].min()
    if CGM_Meal_Datetime >= cgm_df['Date_Time'].min() + pd.to_timedelta(30, unit='m') and CGM_Meal_Datetime <= cgm_df['Date_Time'].max() - pd.to_timedelta(120, unit='m'):
#         CGM_Meal_Date_Time.append(CGM_Meal_Datetime)
        l.append(CGM_Meal_Datetime)
        l.append(insulin_Meal_df.loc[insulin_Meal_df['Date_Time']==datetime]['Ground Truth Label'].iloc[0])
        CGM_Meal_Date_Time.append(l)
# CGM_Meal_Date_Time

mealDataMatrix = []
for mealDateTime in CGM_Meal_Date_Time:
#     print(mealDateTime)
#     idx = cgm_df[cgm_df['Date_Time'] == mealDateTime]['Sensor Glucose (mg/dL)'].index[0]
    idx = cgm_df[cgm_df['Date_Time'] == mealDateTime[0]]['Sensor Glucose (mg/dL)'].index[0]
    l = list(cgm_df['Sensor Glucose (mg/dL)'].iloc[cgm_df.shape[0]-1-idx-6:cgm_df.shape[0]-1-idx+24].values)
    l.append(mealDateTime[1])
    mealDataMatrix.append(l)
#     mealDataMatrix.append(list(cgm_df['Sensor Glucose (mg/dL)'].iloc[cgm_df.shape[0]-1-idx-6:cgm_df.shape[0]-1-idx+24].values))
# mealDataMatrix

meal_df = pd.DataFrame(mealDataMatrix)
meal_df.rename(columns ={30:'Class Label'}, inplace=True)
meal_df = meal_df.dropna()

meal_df['Min_ind'] = meal_df.iloc[:,0:29].idxmin(axis=1)
meal_df['Max_ind'] = meal_df.iloc[:,0:29].idxmax(axis=1)
meal_df['Max_Min_inddiff'] = meal_df['Max_ind'] - meal_df['Min_ind']
meal_df['Max_CGM'] = meal_df.iloc[:,0:29].max(axis=1)
meal_df['Min_CGM'] = meal_df.iloc[:,0:29].min(axis=1)
meal_df['Mean_CGM'] = meal_df.iloc[:,0:29].mean(axis=1)
meal_df['Max_Min_PCT'] = (meal_df['Max_CGM'] - meal_df['Min_CGM'])/meal_df['Min_CGM']*100
meal_df['Max_Min_Slope'] = (meal_df['Max_CGM'] - meal_df['Min_CGM'])/meal_df['Max_Min_inddiff'].abs()
meal_df['Feature1'] =((meal_df['Max_CGM']-meal_df['Mean_CGM'])*(meal_df['Min_CGM']-meal_df['Mean_CGM']))*((meal_df['Max_CGM']-meal_df['Mean_CGM'])*(meal_df['Min_CGM']-meal_df['Mean_CGM']))/meal_df['Min_CGM']

FFT_Min, FFT_Max, FFT_Varr = [], [], []
rff = rfft(meal_df.iloc[:,0:29])
for i in range(len(rff)):
    m = min(rff[i])
    ma = max(rff[i])
    variance = np.var(rff[i])
    FFT_Min.append(m)
    FFT_Max.append(ma)
    FFT_Varr.append(variance)
meal_df["FFTMAX"] = FFT_Max
meal_df["FFTMIN"] = FFT_Min
meal_df["FFTVAR"] = FFT_Varr

# meal_df
# 92%
# meal_df = meal_df[['Mean_CGM','Feature1','Max_Min_PCT','Max_Min_Slope','Class Label']]
# 94%
# meal_df = meal_df[['Feature1','Max_Min_Slope','Class Label']]
# 92%
# meal_df = meal_df[['Max_Min_PCT','Max_Min_Slope','Class Label']]
#94%
# meal_df = meal_df[['Max_Min_Slope','Class Label']]

meal_df = meal_df[['Feature1','Max_CGM','Min_CGM','Class Label']]

X = np.array(meal_df.drop(['Class Label'],1))
X = preprocessing.scale(X)
y = np.array(meal_df['Class Label'])

max1 = max2 = 0
for i in range(10):
    km =KMeans(n_clusters=nBins )
    km.fit(X)
    y_label_km = km.labels_
    accuracy_km = metrics.accuracy_score(y, y_label_km)
    km_sse=km.inertia_
    SSE_score_km = mean_squared_error(y, y_label_km)
    # print(km_sse,SSE_score_km)
    if accuracy_km > max1:
        max1 = accuracy_km
        aril = metrics.adjusted_rand_score(y, y_label_km)
        SSE_score_km = mean_squared_error(y, y_label_km)
        
conv = metrics.cluster.contingency_matrix(y, y_label_km)

total = len(y)
entropy_km= 0
for x in range(len(conv)):
    local_total = sum(conv[x])
    local_entropy = 0
    for i in range(len(conv[x])):
        if conv[x][i] == 0:
            continue
        local_entropy = local_entropy - ((conv[x][i]/local_total)*math.log((conv[x][i]/local_total), 10))
    entropy_km = entropy_km + (local_entropy*(local_total/total))

purity_km =  np.sum(np.amax(conv, axis=0)) / np.sum(conv)

data = StandardScaler().fit_transform(X)
# print(data)
db = DBSCAN(eps = 0.05, min_samples = 1).fit(X)
y_label_db = db.labels_
n_clusters_ = len(set(y_label_db)) - (1 if -1 in y_label_db else 0)
accuracy_db = metrics.accuracy_score(y, y_label_db)
cluster_score = metrics.adjusted_rand_score(y, y_label_db)
if accuracy_db > max2:
    max2 = accuracy_db
    SSE_score_db = mean_squared_error(y, y_label_km)

conv = metrics.cluster.contingency_matrix(y, y_label_db)
total = len(y)
entropy_db= 0
for x in range(len(conv)):
    local_total = sum(conv[x])
    local_entropy = 0
    for i in range(len(conv[x])):
        if conv[x][i] == 0:
            continue
        local_entropy = local_entropy - ((conv[x][i]/local_total)*math.log((conv[x][i]/local_total), 2))
    entropy_db = entropy_km + (local_entropy*(local_total/total))
purity_db =  np.sum(np.amax(conv, axis=0)) / np.sum(conv)

np_array = np.array([SSE_score_km, SSE_score_db, entropy_km, entropy_db, purity_km, purity_db])

np_array.tofile('Result.csv', sep = ',')       