# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 21:54:58 2021

@author: Weizhou Kong
"""

import pandas as pd


CGMData_df = pd.read_csv('CGMData.csv', parse_dates=[['Date', 'Time']],index_col='Date_Time', low_memory=False)
InsulinData_df = pd.read_csv('InsulinData.csv', parse_dates=[['Date', 'Time']], index_col='Date_Time', low_memory=False)

# Reverse dataframe to straight the date time order
CGMData_df = CGMData_df.iloc[::-1]
InsulinData_df = InsulinData_df.iloc[::-1]

# Find auto mode start date
filt = InsulinData_df['Alarm'] =='AUTO MODE ACTIVE PLGM OFF'
AutoModeStartDate = InsulinData_df.loc[filt].index[0]

# Split CGMData to Auto mode and Manual Mode
CGM_Manual_df = CGMData_df.loc[CGMData_df.index < AutoModeStartDate][['Sensor Glucose (mg/dL)', 'ISIG Value']]
CGM_Auto_df = CGMData_df.loc[CGMData_df.index >= AutoModeStartDate][['Sensor Glucose (mg/dL)', 'ISIG Value']]

# Remove Date_Time index 
CGM_Auto_df.reset_index(inplace=True)
CGM_Manual_df.reset_index(inplace=True)

# Remove missing data from 'Sensor Glucose (mg/dL)' column
CGM_Auto_df.dropna(subset=['Sensor Glucose (mg/dL)'],inplace=True)
CGM_Manual_df.dropna(subset=['Sensor Glucose (mg/dL)'],inplace=True)

# Group data by date
CGM_Auto_df_grp_date = CGM_Auto_df.groupby(CGM_Auto_df['Date_Time'].dt.date)
CGM_Manual_df_grp_date = CGM_Manual_df.groupby(CGM_Manual_df['Date_Time'].dt.date)

# Remove daily data count less than 80% and greater than 288
wholeday_Auto_df = CGM_Auto_df_grp_date.filter(lambda x: (x['Sensor Glucose (mg/dL)'].count()>=230) & (x['Sensor Glucose (mg/dL)'].count()<=288))
wholeday_Manual_df = CGM_Manual_df_grp_date.filter(lambda x: (x['Sensor Glucose (mg/dL)'].count()>=230) & (x['Sensor Glucose (mg/dL)'].count()<=288))

wholeday_Auto_df_grp = wholeday_Auto_df.groupby(wholeday_Auto_df['Date_Time'].dt.date)
wholeday_Manual_df_grp = wholeday_Manual_df.groupby(wholeday_Manual_df['Date_Time'].dt.date)

totalAutoModeDays = wholeday_Auto_df_grp.count().shape[0]
totalManualModeDays = wholeday_Manual_df_grp.count().shape[0]
totalDays = totalAutoModeDays + totalManualModeDays

wholeday_Auto_PercentSum = [0, 0, 0, 0, 0, 0] 
wholeday_Manual_PercentSum = [0, 0, 0, 0, 0, 0]
overnight_Auto_PercentSum = [0, 0, 0, 0, 0, 0] 
overnight_Manual_PercentSum = [0, 0, 0, 0, 0, 0] 
day_Auto_PercentSum = [0, 0, 0, 0, 0, 0]
day_Manual_PercentSum = [0, 0, 0, 0, 0, 0]

for name, group in wholeday_Auto_df_grp:

    wholeday_Auto_PercentSum[0] += (group[group['Sensor Glucose (mg/dL)']>180].count()[1])*100/288
    wholeday_Auto_PercentSum[1] += (group[group['Sensor Glucose (mg/dL)']>250].count()[1])*100/288
    wholeday_Auto_PercentSum[2] += (group[(group['Sensor Glucose (mg/dL)']>=70) & (group['Sensor Glucose (mg/dL)']<=180)].count()[1])*100/288
    wholeday_Auto_PercentSum[3] += (group[(group['Sensor Glucose (mg/dL)']>=70) & (group['Sensor Glucose (mg/dL)']<=150)].count()[1])*100/288
    wholeday_Auto_PercentSum[4] += (group[group['Sensor Glucose (mg/dL)']<70].count()[1])*100/288
    wholeday_Auto_PercentSum[5] += (group[group['Sensor Glucose (mg/dL)']<54].count()[1])*100/288
                                          
    overnight_Auto_df = group.loc[(group['Date_Time'].dt.time>=pd.to_datetime('00:00:00').time()) & (group['Date_Time'].dt.time<pd.to_datetime('06:00:00').time())]
    overnight_Auto_PercentSum[0] += (overnight_Auto_df[overnight_Auto_df['Sensor Glucose (mg/dL)']>180].count()[1])*100/288
    overnight_Auto_PercentSum[1] += (overnight_Auto_df[overnight_Auto_df['Sensor Glucose (mg/dL)']>250].count()[1])*100/288
    overnight_Auto_PercentSum[2] += (overnight_Auto_df[(overnight_Auto_df['Sensor Glucose (mg/dL)']>=70) & (overnight_Auto_df['Sensor Glucose (mg/dL)']<=180)].count()[1])*100/288
    overnight_Auto_PercentSum[3] += (overnight_Auto_df[(overnight_Auto_df['Sensor Glucose (mg/dL)']>=70) & (overnight_Auto_df['Sensor Glucose (mg/dL)']<=150)].count()[1])*100/288
    overnight_Auto_PercentSum[4] += (overnight_Auto_df[overnight_Auto_df['Sensor Glucose (mg/dL)']<70].count()[1])*100/288
    overnight_Auto_PercentSum[5] += (overnight_Auto_df[overnight_Auto_df['Sensor Glucose (mg/dL)']<54].count()[1])*100/288
    
    day_Auto_df = group.loc[(group['Date_Time'].dt.time>=pd.to_datetime('06:00:00').time()) & (group['Date_Time'].dt.time<pd.to_datetime('23:59:59').time())]
    day_Auto_PercentSum[0] += (day_Auto_df[day_Auto_df['Sensor Glucose (mg/dL)']>180].count()[1])*100/288
    day_Auto_PercentSum[1] += (day_Auto_df[day_Auto_df['Sensor Glucose (mg/dL)']>250].count()[1])*100/288
    day_Auto_PercentSum[2] += (day_Auto_df[(day_Auto_df['Sensor Glucose (mg/dL)']>=70) & (day_Auto_df['Sensor Glucose (mg/dL)']<=180)].count()[1])*100/288
    day_Auto_PercentSum[3] += (day_Auto_df[(day_Auto_df['Sensor Glucose (mg/dL)']>=70) & (day_Auto_df['Sensor Glucose (mg/dL)']<=150)].count()[1])*100/288
    day_Auto_PercentSum[4] += (day_Auto_df[day_Auto_df['Sensor Glucose (mg/dL)']<70].count()[1])*100/288
    day_Auto_PercentSum[5] += (day_Auto_df[day_Auto_df['Sensor Glucose (mg/dL)']<54].count()[1])*100/288
    
    
for name, group in wholeday_Manual_df_grp:

    wholeday_Manual_PercentSum[0] += (group[group['Sensor Glucose (mg/dL)']>180].count()[1])*100/288
    wholeday_Manual_PercentSum[1] += (group[group['Sensor Glucose (mg/dL)']>250].count()[1])*100/288
    wholeday_Manual_PercentSum[2] += (group[(group['Sensor Glucose (mg/dL)']>=70) & (group['Sensor Glucose (mg/dL)']<=180)].count()[1])*100/288
    wholeday_Manual_PercentSum[3] += (group[(group['Sensor Glucose (mg/dL)']>=70) & (group['Sensor Glucose (mg/dL)']<=150)].count()[1])*100/288
    wholeday_Manual_PercentSum[4] += (group[group['Sensor Glucose (mg/dL)']<70].count()[1])*100/288
    wholeday_Manual_PercentSum[5] += (group[group['Sensor Glucose (mg/dL)']<54].count()[1])*100/288
                                          
    overnight_Manual_df = group.loc[(group['Date_Time'].dt.time>=pd.to_datetime('00:00:00').time()) & (group['Date_Time'].dt.time<pd.to_datetime('06:00:00').time())]
    overnight_Manual_PercentSum[0] += (overnight_Manual_df[overnight_Manual_df['Sensor Glucose (mg/dL)']>180].count()[1])*100/288
    overnight_Manual_PercentSum[1] += (overnight_Manual_df[overnight_Manual_df['Sensor Glucose (mg/dL)']>250].count()[1])*100/288
    overnight_Manual_PercentSum[2] += (overnight_Manual_df[(overnight_Manual_df['Sensor Glucose (mg/dL)']>=70) & (overnight_Manual_df['Sensor Glucose (mg/dL)']<=180)].count()[1])*100/288
    overnight_Manual_PercentSum[3] += (overnight_Manual_df[(overnight_Manual_df['Sensor Glucose (mg/dL)']>=70) & (overnight_Manual_df['Sensor Glucose (mg/dL)']<=150)].count()[1])*100/288
    overnight_Manual_PercentSum[4] += (overnight_Manual_df[overnight_Manual_df['Sensor Glucose (mg/dL)']<70].count()[1])*100/288
    overnight_Manual_PercentSum[5] += (overnight_Manual_df[overnight_Manual_df['Sensor Glucose (mg/dL)']<54].count()[1])*100/288
    
    day_Manual_df = group.loc[(group['Date_Time'].dt.time>=pd.to_datetime('06:00:00').time()) & (group['Date_Time'].dt.time<pd.to_datetime('23:59:59').time())]
    day_Manual_PercentSum[0] += (day_Manual_df[day_Manual_df['Sensor Glucose (mg/dL)']>180].count()[1])*100/288
    day_Manual_PercentSum[1] += (day_Manual_df[day_Manual_df['Sensor Glucose (mg/dL)']>250].count()[1])*100/288
    day_Manual_PercentSum[2] += (day_Manual_df[(day_Manual_df['Sensor Glucose (mg/dL)']>=70) & (day_Manual_df['Sensor Glucose (mg/dL)']<=180)].count()[1])*100/288
    day_Manual_PercentSum[3] += (day_Manual_df[(day_Manual_df['Sensor Glucose (mg/dL)']>=70) & (day_Manual_df['Sensor Glucose (mg/dL)']<=150)].count()[1])*100/288
    day_Manual_PercentSum[4] += (day_Manual_df[day_Manual_df['Sensor Glucose (mg/dL)']<70].count()[1])*100/288
    day_Manual_PercentSum[5] += (day_Manual_df[day_Manual_df['Sensor Glucose (mg/dL)']<54].count()[1])*100/288
    
result = {
    
'1':[overnight_Manual_PercentSum[0]/totalDays,overnight_Auto_PercentSum[0]/totalDays],
    
'2':[overnight_Manual_PercentSum[1]/totalDays,overnight_Auto_PercentSum[1]/totalDays],
    
'3':[overnight_Manual_PercentSum[2]/totalDays,overnight_Auto_PercentSum[2]/totalDays],

'4':[overnight_Manual_PercentSum[3]/totalDays,overnight_Auto_PercentSum[3]/totalDays],
    
'5':[overnight_Manual_PercentSum[4]/totalDays,overnight_Auto_PercentSum[4]/totalDays],

'6':[overnight_Manual_PercentSum[5]/totalDays,overnight_Auto_PercentSum[5]/totalDays],

    
'7':[day_Manual_PercentSum[0]/totalDays,day_Auto_PercentSum[0]/totalDays],
    
'8':[day_Manual_PercentSum[1]/totalDays,day_Auto_PercentSum[1]/totalDays],
    
'9':[day_Manual_PercentSum[2]/totalDays,day_Auto_PercentSum[2]/totalDays],

'10':[day_Manual_PercentSum[3]/totalDays,day_Auto_PercentSum[3]/totalDays],

'11':[day_Manual_PercentSum[4]/totalDays,day_Auto_PercentSum[4]/totalDays],

'12':[day_Manual_PercentSum[5]/totalDays,day_Auto_PercentSum[5]/totalDays],


'13':[wholeday_Manual_PercentSum[0]/totalDays,wholeday_Auto_PercentSum[0]/totalDays],

'14':[wholeday_Manual_PercentSum[1]/totalDays,wholeday_Auto_PercentSum[1]/totalDays],
   
'15':[wholeday_Manual_PercentSum[2]/totalDays,wholeday_Auto_PercentSum[2]/totalDays],

'16':[wholeday_Manual_PercentSum[3]/totalDays,wholeday_Auto_PercentSum[3]/totalDays],

'17':[wholeday_Manual_PercentSum[4]/totalDays,wholeday_Auto_PercentSum[4]/totalDays],

'18':[wholeday_Manual_PercentSum[5]/totalDays,wholeday_Auto_PercentSum[5]/totalDays],

}

result_df = pd.DataFrame(result, index =['Manual Mode', 'Auto Mode'])
result_df
result_df.to_csv('Results.csv', index=False, header=False)