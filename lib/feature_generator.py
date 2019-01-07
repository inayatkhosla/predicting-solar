"""
generation_parser.py:
    Generates trailing features. Processes and combines 
    actuals, capacity, and weather data into features for
    modeling 

    TrailingFeatureGenerator (cls): Generates trailing features
    FeatureGenerator (cls): Combines data and creates final features
"""


import pathlib
import pandas as pd
import numpy as np
import yaml


class TrailingFeatureGenerator(object):
    """
    Generates features detailing trailing solar power 
    production

    Args:
        horizon (num): Number of hours to lag trails by
        actuals (df): Actual generation
        int_lengths (dict): Data interval lengths by operator
        hourly (bool): Flag for hourly data
    """
    def __init__(self, horizon, actuals, int_lengths, hourly):
        self.horizon = horizon
        self.actuals = actuals
        self.int_lengths = int_lengths
        self.hourly = hourly
        


    def get_current_trails(self, df, lag):
        df['per_bef'] = df['solar'].shift(lag).fillna(0)
        df['per_bef_2'] = df['solar'].shift(lag+1).fillna(0)
        df['per_bef_3'] = df['solar'].shift(lag+2).fillna(0)
        df['per_bef_roc'] = (df['per_bef'] - df['per_bef_3'])/df['per_bef_3']
        df['per_bef_roc'] = df['per_bef_roc'].fillna(0).replace(np.inf, 100)
        return df
    
    
    def get_current_generation_stats(self, df, num_hours, int_length):
        window = int((num_hours * 60)/int_length)
        prefix = str(num_hours) + 'h_'
        df['mean'] = df['per_bef'].rolling(window, min_periods=1).mean()
        df['median'] = df['per_bef'].rolling(window, min_periods=1).median()
        df['min'] = df['per_bef'].rolling(window, min_periods=1).min()
        df['max'] = df['per_bef'].rolling(window, min_periods=1).max()
        statcols = ['mean','median','min','max']
        df.columns = [prefix + col if col in statcols else col for col in df.columns]
        return df
    
    
    def get_day_trails(self, df, int_length):
        day_length = int((60*24)/int_length)
        df['day_bef'] = df['solar'].shift(day_length)
        df['day_bef_2'] = df['solar'].shift(day_length*2)
        df['day_bef_3'] = df['solar'].shift(day_length*3)
        df = df.dropna().copy()
        df['day_bef_roc'] = (df['day_bef'] - df['day_bef_2'])/df['day_bef_2']
        df['day_bef_roc'] = df['day_bef_roc'].fillna(0).replace(np.inf, 100)
        return df
    
    
    #Iffy, too few values, get rid if you can
    def get_multiday_stats(self, df):
        mdcols = ['day_bef', 'day_bef_2', 'day_bef_3']
        df['multiday_mean'] = df[mdcols].mean(axis=1)
        df['multiday_median'] = df[mdcols].median(axis=1)
        df['multiday_min'] = df[mdcols].min(axis=1)
        df['multiday_max'] = df[mdcols].max(axis=1)
        return df

    
    def generate_trails(self):
        df = self.actuals[['int_start','operator','solar']].copy()
        trl_fts = []
        for o in df['operator'].unique():
            if self.hourly:
                int_length = 60
            else:
                int_length = self.int_lengths[o]
            lag = int((self.horizon * 60)/int_length)
            dfo = df[df['operator'] == o].copy()
            dfo = self.get_current_trails(dfo, lag)
            dfo = self.get_day_trails(dfo, int_length)
            dfo = self.get_current_generation_stats(dfo, 1, int_length)
            dfo = self.get_current_generation_stats(dfo, 3, int_length)
            dfo = self.get_multiday_stats(dfo)
            trl_fts.append(dfo)
        trail_features = pd.concat(trl_fts, sort=True)
        trail_features.drop('solar', axis=1, inplace=True)
        return trail_features


class FeatureGenerator(object):
    """
    Processes and combines different datasets into features
    for modeling

    Args:
        operators (list): Operators to generate features for
        rectype (str): 'hist' or 'fc
        horizon (num): Number of hours to lag trails by
        int_lengths (dict): Data interval lengths by operator
        hourly (bool): Flag for hourly data
        datapath (str): Base data folder
    """
    def __init__(self, operators, rectype, horizon=1, interval_lengths='int_lengths.yaml', hourly=False, datapath='data'):
        self.operators = operators
        self.rectype = rectype
        self.horizon = horizon
        self.interval_lengths = interval_lengths
        self.hourly = hourly
        self.datapath = datapath
        
        
        
    def set_filepaths(self):
        Data = pathlib.Path(self.datapath)
        self.in_path = (Data/'processed_data/')
        self.out_path = (Data/'processed_data/')
        self.out_path.mkdir(parents=True, exist_ok=True)
        
        
    def get_interval_lengths(self):
        with open((self.in_path/self.interval_lengths), 'r') as file:
            self.int_lengths = yaml.load(file)
    
    
    def process_capacity(self):
        self.capacity = pd.read_csv((self.in_path/'capacity.csv'))
    
    
    def process_actuals(self):
        if self.hourly:
            actuals = pd.read_csv((self.in_path/'actuals_hourly.csv'))
        else:
            actuals = pd.read_csv((self.in_path/'actuals.csv'))
        self.actuals = actuals[actuals['operator'].isin(self.operators)]
        self.actuals['month_year'] = self.actuals['month_year'].replace(to_replace=r'-', value='', regex=True)
        self.actuals['month_year'] = self.actuals['month_year'].astype(int)
        self.actuals = self.actuals[~self.actuals['solar'].isin(['n/e','-','e"'])]
        self.actuals = self.actuals[~self.actuals['solar'].isnull()]
        self.actuals['solar'] = pd.to_numeric(self.actuals['solar'])
       
        
        
    def process_daily_weather(self):
        dpath = (self.in_path/'daily_weather_{}.csv'.format(self.rectype))
        daily = pd.read_csv(dpath)
        daily['date'] = pd.to_datetime(daily['localtime']).dt.date
        daily.drop(['localtime','time'], axis=1, inplace=True)
        daily.columns = ['d_' + str(col) if col not in ['operator','date'] else col for col in daily.columns]
        return daily
    
    
    def process_hourly_weather(self):
        hpath = (self.in_path/'hourly_weather_{}.csv'.format(self.rectype))
        hourly = pd.read_csv(hpath)
        hourly['date'] = pd.to_datetime(hourly['localtime']).dt.date
        hourly.columns = ['h_' + str(col) if col not in ['operator','date'] else col for col in hourly.columns]
        return hourly
    
    
    def process_weather(self):
        daily = self.process_daily_weather()
        hourly = self.process_hourly_weather()
        self.weather = daily.merge(hourly, on=['operator','date'])
        
        
    def add_trailing_features(self):
        tfg = TrailingFeatureGenerator(self.horizon, self.actuals, self.int_lengths, self.hourly)
        self.trail_features = tfg.generate_trails()
        
    
    def combine(self):
        model_data = self.actuals.merge(self.capacity, on = ['operator','year'])
        model_data = model_data.merge(self.trail_features, on=['operator','int_start'])
        model_data = model_data.merge(self.weather, left_on=['operator','base_hour'], right_on=['operator','h_localtime'])
        model_data = model_data.dropna()
        self.model_data = model_data
        
        
    def filter_nightime(self):
       self.model_data = self.model_data[(self.model_data['hour'] > 5) & (self.model_data['hour'] < 21)]
       self.model_data = self.model_data.reset_index(drop=True)
   
    
    def clean(self):
        remcols = 'pulltime|localtime|date|week_year|base_hour'
        cols_to_rmv = self.model_data.columns.str.contains(r'{}'.format(remcols), case=False)
        self.model_data = self.model_data[self.model_data.columns[~cols_to_rmv]].copy()
        

    def write(self):
        if self.hourly:
            fname = 'model_data_hourly_{}'.format(self.horizon)
        else:
            fname = 'model_data_{}'.format(self.horizon)
        self.model_data.to_feather((self.out_path/'{}'.format(fname)))
        
    
    def generate(self):
        self.set_filepaths()
        self.get_interval_lengths()
        self.process_capacity()
        self.process_actuals()
        self.process_weather() 
        self.add_trailing_features()
        self.combine()
        self.filter_nightime()
        self.clean()
        self.write()