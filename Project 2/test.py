# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 00:20:32 2021

@author: wayne
"""

import pandas as pd
import pickle

testSet = pd.read_csv("test.csv", error_bad_lines=False)
# testSet
testSet['Max_CGM'] = testSet.max(axis=1)
testSet['Min_CGM'] = testSet.min(axis=1)
testSet['Mean_CGM'] = testSet.mean(axis=1)
testSet['Max_Min_PCT'] = (testSet['Max_CGM'] - testSet['Min_CGM'])/testSet['Min_CGM']*100
testSet['Feature1'] =((testSet['Max_CGM']-testSet['Mean_CGM'])*(testSet['Min_CGM']-testSet['Mean_CGM']))*((testSet['Max_CGM']-testSet['Mean_CGM'])*(testSet['Min_CGM']-testSet['Mean_CGM']))/testSet['Min_CGM']
testSet = testSet[['Feature1','Max_Min_PCT']]

clf = pickle.load(open('71_decision_tree.pkl', 'rb'))
# clf = pickle.load(open('79_SVM.pkl', 'rb'))
# clf = pickle.load(open('SVM.pkl', 'rb'))
y_pred = clf.predict(testSet)
new_y_pred = []
for i in y_pred:
    if(i > 0):
        new_y_pred.append(1)
    else:
        new_y_pred.append(0)
result = pd.DataFrame(new_y_pred)
result.to_csv('Result.csv',index=False)

