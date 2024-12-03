# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:14:46 2024

@author: olivi
"""

from process_data import process_data
from sklearn import linear_model
from sklearn.feature_selection import SelectPercentile, f_classif, VarianceThreshold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector


def logistic_regression():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = process_data()
    del X_train['snow']
    del X_test['snow']
    print(X_train.shape)
    print(X_train.columns)
   
    
    # logr = linear_model.LogisticRegression()
    # logr.fit(X_train,Y_train)
    # print(logr.get_params())
    
    # predict_prob = logr.predict_proba(X_test)
    # print(logr.classes_)
    # print(predict_prob[0:5])
    
    # Perform feature selection
    select_percentile = SelectPercentile(score_func=f_classif,percentile=20)
    X_new = select_percentile.fit_transform(X_train, Y_train)
    print("Shape after SelectPercentile:", X_new.shape)
    # Get the selected feature names
    selected_feature_names = X_train.columns[select_percentile.get_support()]

    # Create a DataFrame for the transformed X_test
    X_test_new = pd.DataFrame(X_test, columns=selected_feature_names)
    print(X_test_new.head())
    
    logr = linear_model.LogisticRegression()
    logr.fit(X_new,Y_train)
    print(logr.get_params())
    
    predict_prob = logr.predict_proba(X_test_new)
    print(logr.classes_)
    print(predict_prob[0:50:5])
    
    Y_pred = classify(logr,predict_prob)
    print(Y_pred[0:50:5])
   # a = get_accuracy(Y_test, Y_pred)
    #print(Y_test)
    #print(a)
    
def classify(model,predict_prob):
    if model.classes_[0] == 'high_bike_demand':
        high, low = 1,0
    else: 
        high,low = 0,1
    result = []
    for val in predict_prob:
        if val[0] >= 0.5:
            result.append(high)
        else:
           result.append(low) 
    return np.array(result)
    
def get_accuracy(Y_true,Y_pred):
    return(accuracy_score(Y_true,Y_pred))


logistic_regression()
    
def forward_selection(X, y, threshold_p=0.05, threshold_corr=0.9):
    selected_features = []
    remaining_features = list(range(X.shape[1]))
    correlations = X.corr().abs()
    while remaining_features:
        scores = []
        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            X_subset = X[:, features_to_test]
            
            # Fitting a model and computing F-score and p-value
            fscore, _ = f_classif(X_subset, y)
            model = sm.Logit(y, sm.add_constant(X_subset)).fit(disp=0)
            pvalue = model.pvalues[1:]  # Skip the constant
            
            if all(p <= threshold_p for p in pvalue) and np.mean(fscore) > 0:
                scores.append((np.mean(fscore), feature))
        
        if not scores:
            break
        
        scores.sort(reverse=True)  # Sort by F-score descending
        best_feature = scores[0][1]
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        # Check correlation
        for f1 in selected_features:
            for f2 in selected_features:
                if f1 != f2 and correlations[f1][f2] > threshold_corr:
                    if f2 in selected_features:
                        selected_features.remove(f2)
    
    return selected_features












