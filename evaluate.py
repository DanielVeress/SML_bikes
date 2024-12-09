from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.dummy import DummyClassifier
from utils.constants import SEED


def evaluate_pred(y_pred:Series, y_true:Series):
    '''Evaluate prediction and calculate metrics'''

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError('The prediction and the true labels have different number of rows!')
    
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['mix'] = metrics['accuracy']*metrics['recall']*metrics['precision']*10

    return metrics


def print_dummy_evaluations(X_train:DataFrame, Y_train:Series, X_test:DataFrame, Y_test:Series) -> None:
    '''Evaluates dummy models'''

    dummy_strategies = ['most_frequent', 'stratified', 'uniform']
    for strat in dummy_strategies:
        dummy_model = DummyClassifier(strategy=strat, random_state=SEED)
        
        dummy_model.fit(X_train, Y_train)
        
        # predict
        y_pred = dummy_model.predict(X_test)
        y_true = Y_test

        # evaluate
        metrics = evaluate_pred(y_pred, y_true)
        print(f'\n{strat.capitalize()}')
        for metric, value in metrics.items():
            print(f'  {metric.capitalize()}: \t{value}')