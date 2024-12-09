# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:14:46 2024

@author: olivi
"""
import sys
sys.path.append("..")

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
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

from evaluate import evaluate_pred
from visualization import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

import statistics
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def analyze_features(X_train,best_model):
    coefficients = best_model.named_steps['logisticregression'].coef_.flatten()  # Flatten in case of binary classification
    feature_names = X_train.columns
    # Create a DataFrame
    feature_strength = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    # Sort by absolute coefficient value
    feature_strength['AbsCoefficient'] = feature_strength['Coefficient'].abs()
    feature_strength = feature_strength.sort_values(by='AbsCoefficient', ascending=False)

    # Print feature strength
    print(feature_strength)
    


def logistic_regression():
   
    X_train, X_test,  Y_train, Y_test= process_data({"train": 0.8, "test": 0.2}, is_random=False)
    Y_train = np.array(Y_train).ravel()

    # Create a pipeline with scaler and logistic regression
    pipe = make_pipeline(StandardScaler(), linear_model.LogisticRegression(max_iter=1000, solver='saga', tol=0.1))
     
    # Create a parameter grid
    param_grid = {
        'logisticregression__C': [0.1, 1, 10, 100],
        'logisticregression__penalty': ['l1', 'l2']
    }
     
    cv = StratifiedKFold(n_splits=50, shuffle=True)
    grid_search = GridSearchCV(pipe, param_grid, cv=cv,scoring='accuracy', n_jobs=-1)
    
    Y_train =  np.array(Y_train).ravel()
    # Fit the model
    grid_search.fit(X_train, Y_train)

    # Get the best parameters and the best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"Best parameters: {best_params}")
    
    print(best_model)
    analyze_features(X_train,best_model)
    

    # Evaluate the best model on the test set
    predict_prob = best_model.predict_proba(X_test)
    Y_pred = classify(best_model, predict_prob)

    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    
    print("Acc training: ", accuracy_score(best_model.predict(X_train), Y_train))  
    print("Acc test: ", get_accuracy(Y_pred, Y_test))  

    print(evaluate_pred(Y_pred, Y_test))
    return evaluate_pred(Y_pred, Y_test)




  #  print(Y_pred[0:50:5])
    #a = get_accuracy(Y_test, Y_pred)
    #print(Y_test)
   # print(a)
 #   print(evaluate_pred(Y_pred, Y_test))
    # Confusion matrix\n
  #  tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
  #  confusion_matrix_data = np.array([[tn, fp], [fn, tp]])
  #  plot_confusion_matrix(confusion_matrix_data)
  #  "\n",
  #  print(f'True Positive (TP): {tp}')
  #  print(f'True Negative (TN): {tn}')
  #  print(f'False Positive (FP): {fp}')
  #  print(f'False Negative (FN): {fn}')
  
  #  print("Acc training: ", accuracy_score(logr.predict(X_train), Y_train))  
   # print("Acc test: ", get_accuracy(Y_pred, Y_test))  
  #  return evaluate_pred(Y_pred, Y_test)
   
 


        
def classify(model,predict_prob):
    if model.classes_[0] == '1':
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





def meassure_model(n=100):
    stats = {"accuracy": [],"recall":[],"precision":[]}
    
    for i in range(n):
        print(i)
        pred = logistic_regression()
        a = pred["accuracy"]
        r = pred["recall"]
        p = pred["precision"]
        
        stats["accuracy"].append(a)
        stats["recall"].append(r)
        stats["precision"].append(p)
        
    for key,val in stats.items():
        print(f'{key}: {statistics.mean(val)}, {statistics.variance(val)}')
        
            

if __name__ == "__main__":
    logistic_regression()


