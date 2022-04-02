# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 00:08:30 2021

@author: wayne
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
import pickle

#load Insulin and CGM Data
insulin_df = pd.read_csv('InsulinData.csv', parse_dates=[['Date','Time']],low_memory=False).iloc[::-1]
cgm_df = pd.read_csv('CGMData.csv', parse_dates=[['Date','Time']],low_memory=False).iloc[::-1]

insulin_df2 = pd.read_csv('Insulin_patient2.csv', parse_dates=[['Date','Time']],low_memory=False).iloc[::-1]
cgm_df2 = pd.read_csv('CGM_patient2.csv', parse_dates=[['Date','Time']],low_memory=False).iloc[::-1]

insulin_df = insulin_df[['Date_Time','BWZ Carb Input (grams)']]
insulin_df2 = insulin_df2[['Date_Time','BWZ Carb Input (grams)']]

#filter NaNs and Zeros
insulin_Meal_df= insulin_df[insulin_df['BWZ Carb Input (grams)'].notnull() & insulin_df['BWZ Carb Input (grams)'] != 0]
insulin_Meal_df2= insulin_df2[insulin_df2['BWZ Carb Input (grams)'].notnull() & insulin_df2['BWZ Carb Input (grams)'] != 0]

#mealstartdates from insulin csv
insulin_Meal_df.loc[:,'Time_dif1'] = insulin_Meal_df['Date_Time'].diff(periods=-1)/np.timedelta64(1,'h')
insulin_Meal_df.loc[:,'Time_dif2'] = insulin_Meal_df['Date_Time'].diff(periods=1)/np.timedelta64(1,'h')
insulin_Meal_df2.loc[:,'Time_dif1'] = insulin_Meal_df2['Date_Time'].diff(periods=-1)/np.timedelta64(1,'h')
insulin_Meal_df2.loc[:,'Time_dif2'] = insulin_Meal_df2['Date_Time'].diff(periods=1)/np.timedelta64(1,'h')

filt=((insulin_Meal_df['Time_dif1']<-2) | (insulin_Meal_df['Time_dif2'].iloc[-1:]>2))
filt2=((insulin_Meal_df2['Time_dif1']<-2) | (insulin_Meal_df2['Time_dif2'].iloc[-1:]>2))
insulin_Meal_df = insulin_Meal_df.loc[filt]
insulin_Meal_df2 = insulin_Meal_df2.loc[filt2]

CGM_Meal_Date_Time = []

for datetime in insulin_Meal_df['Date_Time']:
    CGM_Meal_Datetime = cgm_df.loc[cgm_df['Date_Time']>=datetime]['Date_Time'].min()
    if CGM_Meal_Datetime >= cgm_df['Date_Time'].min() + pd.to_timedelta(30, unit='m') and CGM_Meal_Datetime <= cgm_df['Date_Time'].max() - pd.to_timedelta(120, unit='m'):
        CGM_Meal_Date_Time.append(CGM_Meal_Datetime)

CGM_Meal_Date_Time2 = []

for datetime in insulin_Meal_df2['Date_Time']:
    CGM_Meal_Datetime2 = cgm_df2.loc[cgm_df2['Date_Time']>=datetime]['Date_Time'].min()
    if CGM_Meal_Datetime2 >= cgm_df2['Date_Time'].min() + pd.to_timedelta(30, unit='m') and CGM_Meal_Datetime2 <= cgm_df2['Date_Time'].max() - pd.to_timedelta(120, unit='m'):
        CGM_Meal_Date_Time2.append(CGM_Meal_Datetime2)

CGM_No_Meal_Date_Time = []
for i, CGM_Meal_Datetime in enumerate(CGM_Meal_Date_Time[:-1]):
#     print(i, CGM_Meal_Datetime)
    No_Meal_Starttime = CGM_Meal_Datetime + pd.to_timedelta(120, unit='m')
    while No_Meal_Starttime < CGM_Meal_Date_Time[i+1]:
#     while No_Meal_Starttime < CGM_Meal_Date_Time[i+1]-pd.to_timedelta(30, unit='m'):
        CGM_No_Meal_Datetime = cgm_df.loc[cgm_df['Date_Time'] >= No_Meal_Starttime]['Date_Time'].min()
        CGM_No_Meal_Date_Time.append(CGM_No_Meal_Datetime)
        No_Meal_Starttime += pd.to_timedelta(120, unit='m')

CGM_No_Meal_Date_Time2 = []
for i, CGM_Meal_Datetime2 in enumerate(CGM_Meal_Date_Time2[:-1]):
#     print(i, CGM_Meal_Datetime)
    No_Meal_Starttime2 = CGM_Meal_Datetime2 + pd.to_timedelta(120, unit='m')
    while No_Meal_Starttime2 < CGM_Meal_Date_Time2[i+1]:
#     while No_Meal_Starttime < CGM_Meal_Date_Time[i+1]-pd.to_timedelta(30, unit='m'):
        CGM_No_Meal_Datetime2 = cgm_df2.loc[cgm_df2['Date_Time'] >= No_Meal_Starttime2]['Date_Time'].min()
        CGM_No_Meal_Date_Time2.append(CGM_No_Meal_Datetime2)
        No_Meal_Starttime2 += pd.to_timedelta(120, unit='m')

mealDataMatrix = []
for mealDateTime in CGM_Meal_Date_Time:
    idx = cgm_df[cgm_df['Date_Time'] == mealDateTime]['Sensor Glucose (mg/dL)'].index[0]
    mealDataMatrix.append(list(cgm_df['Sensor Glucose (mg/dL)'].iloc[cgm_df.shape[0]-1-idx-6:cgm_df.shape[0]-1-idx+24].values))

mealDataMatrix2 = []
for mealDateTime2 in CGM_Meal_Date_Time2:
    idx = cgm_df2[cgm_df2['Date_Time'] == mealDateTime2]['Sensor Glucose (mg/dL)'].index[0]
    mealDataMatrix2.append(list(cgm_df2['Sensor Glucose (mg/dL)'].iloc[cgm_df2.shape[0]-1-idx-6:cgm_df2.shape[0]-1-idx+24].values))

no_mealDataMatrix = []
for no_mealDateTime in CGM_No_Meal_Date_Time:
    idx = cgm_df[cgm_df['Date_Time'] == no_mealDateTime]['Sensor Glucose (mg/dL)'].index[0]
    no_mealDataMatrix.append(list(cgm_df['Sensor Glucose (mg/dL)'].iloc[cgm_df.shape[0]-1-idx:cgm_df.shape[0]-1-idx+24].values))

no_mealDataMatrix2 = []
for no_mealDateTime2 in CGM_No_Meal_Date_Time2:
    idx = cgm_df2[cgm_df2['Date_Time'] == no_mealDateTime2]['Sensor Glucose (mg/dL)'].index[0]
    no_mealDataMatrix2.append(list(cgm_df2['Sensor Glucose (mg/dL)'].iloc[cgm_df2.shape[0]-1-idx:cgm_df2.shape[0]-1-idx+24].values))

mealDate_dict = {}
for datetime, g_value in zip(CGM_Meal_Date_Time,mealDataMatrix):
    mealDate_dict[datetime] = g_value

mealDate_dict2 = {}
for datetime2, g_value2 in zip(CGM_Meal_Date_Time2,mealDataMatrix2):
    mealDate_dict2[datetime2] = g_value2

no_mealDate_dict = {}
for datetime, g_value in zip(CGM_No_Meal_Date_Time,no_mealDataMatrix):
    no_mealDate_dict[datetime] = g_value

no_mealDate_dict2 = {}
for datetime2, g_value2 in zip(CGM_No_Meal_Date_Time2,no_mealDataMatrix2):
    no_mealDate_dict2[datetime2] = g_value2

mealDataMatrix_df = pd.DataFrame.from_dict(mealDate_dict, orient='index')
mealDataMatrix_df = mealDataMatrix_df.dropna()

mealDataMatrix_df2 = pd.DataFrame.from_dict(mealDate_dict2, orient='index')
mealDataMatrix_df2 = mealDataMatrix_df2.dropna()

no_mealDataMatrix_df = pd.DataFrame.from_dict(no_mealDate_dict, orient='index')
no_mealDataMatrix_df = no_mealDataMatrix_df.dropna()

no_mealDataMatrix_df2 = pd.DataFrame.from_dict(no_mealDate_dict2, orient='index')
no_mealDataMatrix_df2 = no_mealDataMatrix_df2.dropna()

meal_df = pd.concat([mealDataMatrix_df, mealDataMatrix_df2], ignore_index=True, sort=False)
no_meal_df = pd.concat([no_mealDataMatrix_df, no_mealDataMatrix_df2],ignore_index=True, sort=False)


meal_df['Max_CGM'] = meal_df.max(axis=1)
meal_df['Min_CGM'] = meal_df.min(axis=1)
meal_df['Mean_CGM'] = meal_df.mean(axis=1)
meal_df['Max_Min_PCT'] = (meal_df['Max_CGM'] - meal_df['Min_CGM'])/meal_df['Min_CGM']*100
meal_df['Feature1'] =((meal_df['Max_CGM']-meal_df['Mean_CGM'])*(meal_df['Min_CGM']-meal_df['Mean_CGM']))*((meal_df['Max_CGM']-meal_df['Mean_CGM'])*(meal_df['Min_CGM']-meal_df['Mean_CGM']))/meal_df['Min_CGM']

no_meal_df['Max_CGM'] = no_meal_df.max(axis=1)
no_meal_df['Min_CGM'] = no_meal_df.min(axis=1)
no_meal_df['Mean_CGM'] = no_meal_df.mean(axis=1)
no_meal_df['Max_Min_PCT'] = (no_meal_df['Max_CGM'] - no_meal_df['Min_CGM'])/no_meal_df['Min_CGM']*100
no_meal_df['Feature1'] =((no_meal_df['Max_CGM']-no_meal_df['Mean_CGM'])*(no_meal_df['Min_CGM']-no_meal_df['Mean_CGM']))*((no_meal_df['Max_CGM']-no_meal_df['Mean_CGM'])*(no_meal_df['Min_CGM']-no_meal_df['Mean_CGM']))/no_meal_df['Min_CGM']

meal_df['Class']=1
meal_df = meal_df[['Feature1','Max_Min_PCT','Class']]

no_meal_df['Class']=0
no_meal_df = no_meal_df[['Feature1','Max_Min_PCT','Class']]

final_df = pd.concat([meal_df,no_meal_df], ignore_index=True, sort=False)

X = np.array(final_df.drop(['Class'],1))
y = np.array(final_df['Class'])

kf = RepeatedKFold(n_splits=5, n_repeats=5)
# # Decision tree
for X_train, X_test in kf.split(X):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predicted_set = clf.predict(X_test)
    accuracy = int(accuracy_score(y_test, predicted_set)*100)
    filename =str(accuracy)+'_decision_tree.pkl'
    pickle.dump(clf, open(filename,'wb'))

#SVM
# for X_train, X_test in kf.split(X):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)
#     clf = svm.SVC(kernel='rbf',gamma=0.009,C=1)
#     clf.fit(X_train, y_train)
#     predicted_set = clf.predict(X_test)
#     accuracy = int(accuracy_score(y_test, predicted_set)*100)
# #     print(classification_report(y_test, predicted_set))
# #     print(confusion_matrix(y_test, predicted_set))
#       # print(accuracy)
#     # filename =str(accuracy)+'_SVM.pkl'
#     filename ='SVM.pkl'
#     pickle.dump(clf, open(filename,'wb'))


