"""
evaluators.py:
    Convenience functions for model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

def mdape(predictions, targets): 
    """Calculates Median Absolute Percentage Error (MDAPE)"""
    return round(np.median(np.abs((predictions - targets) / targets)) * 100, 2)


def rmse(predictions, targets): 
    """Calculates Root Mean Squared Error (RMSE)"""
    return round(np.sqrt(mean_squared_error(targets, predictions)),2)


def ensemble(preds):
    """Averages predictions to create ensembles"""
    preds['e_rxn'] = preds[['rf','xgb','nn']].mean(axis=1)
    preds['e_rxnk'] = preds[['rf','xgb','nn','knn']].mean(axis=1)
    return preds


def evaluate_predictions(metric, preds, operators, models):
    """Calculates errors across models and operators"""
    p = []
    for o in operators:
        vs = preds.loc[preds['operator']==o]
        op = []
        op.append(o)
        for i in models:
            pr = round(metric(vs[i], vs['solar']),2)
            op.append(pr)
        p.append(op) 
    df = pd.DataFrame(p)
    df.columns = ['operator'] + models
    return df


def get_error_percentages(preds, operator, model):
    """Calculates error percentages"""
    df = preds[preds['operator']==operator]
    df = df.set_index(pd.DatetimeIndex(df['int_start']))
    df['error'] = (df[model] - df['solar'])
    df['error_perc'] = (df['error']/df['solar']) * 100
    return df


def plot_error_distribution(df):
    """Plots error distribution"""
    plt.rcParams['figure.figsize'] = [15, 7]
    print(df['error_perc'].describe())
    df['error_perc'].hist(range=(-50,50), bins=20)
    plt.ticklabel_format(style='plain', axis='y')
    plt.title('Error Distribution')
    plt.xlabel('Percentage Error')
    plt.ylabel('Frequency')
    plt.grid(False)
    sns.despine()


def plot_results(df, model, start, end):
    """Plots Actuals vs Predictions for a given date range"""
    for i in ['solar', model]:
        p = df[start:end][i].plot(label=i)
        p.legend()
    operator = df['operator'].unique()[0]
    plt.ticklabel_format(style='plain', axis='y')
    plt.title('Solar Power Generation Over Time \n Predictions vs Actuals \n {}: {}'.format(operator, start[:7]))
    plt.xlabel('Date')
    plt.ylabel('Solar Power Generation (MWh)')
    plt.legend(['Actuals', 'Predictions'],loc='upper left', frameon=True, fancybox=True, fontsize=14)
    sns.despine()


def get_means(df, operator, start, end, phenomena):
    """Calculates mean value of weather phenomena for a given date range"""
    d = df[df['operator'] == operator]
    d = d.set_index(pd.DatetimeIndex(d['int_start']))
    means = d[start:end].filter(regex=(phenomena)).mean(axis=1)
    return means