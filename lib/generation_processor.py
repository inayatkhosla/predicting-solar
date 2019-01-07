"""
generation_parser.py:
    Cleans and transforms energy generation data
    provided by the ENTSOE Transparency Platform - handles both Actual
    and Forecast generation.

    EnergyDataTransformer (cls): Preprocesses data
    ActualGenerationParser (cls): Processes actual solar power generation
    ForecastGenerationParser (cls): Processes forecast solar power generation
"""


import pathlib, glob
import pandas as pd
import yaml


class EnergyDataTransformer(object):
    def __init__(self):
        pass
    
    
    def set_filepaths(self, gen_type):
        Data = pathlib.Path(self.datapath)
        self.in_path = (Data/'raw_data/generation/{}/'.format(gen_type))
        self.out_path = (Data/'processed_data/')
        self.out_path.mkdir(parents=True, exist_ok=True)
        
    
    def parse_datetimes(self, df):
        s = df['interval'].str.split(' - ')
        df['int_start'] = s.str[0]
        df['int_end'] = s.str[1].map(lambda x: x.rstrip(' (CET)'))
        df[['int_start','int_end']] = df[['int_start','int_end']].apply(pd.to_datetime, format='%d.%m.%Y %H:%M')
        df.drop('interval', axis=1, inplace=True)
        return df
    
    
    def calc_intervals(self, df):
        df['int_length'] = (df['int_end'] - df['int_start']).dt.total_seconds()/60
        intervals = df[['operator','int_length']].drop_duplicates()
        int_lengths = dict(zip(intervals['operator'], intervals['int_length']))
        df.drop(['int_end', 'int_length'], axis=1, inplace=True)
        return int_lengths
    
    
    def convert_to_hourly(self, df, col):
        df = df.set_index(pd.DatetimeIndex(df['int_start']))
        df.drop('int_start', axis=1, inplace=True)
        df['base_hour'] = df.index.floor('H')
        df = df[~df[col].isin(['n/e','-','e"'])]
        df[col] = pd.to_numeric(df[col])
        dfh = df.groupby(['operator','base_hour'])[col].sum().reset_index()
        dfh = dfh.rename(columns={'base_hour': 'int_start'})
        return dfh

        
    def add_datetimecols(self, df):
        df = df.set_index(pd.DatetimeIndex(df['int_start']))
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['week'] = df.index.week
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['base_hour'] = df.index.floor('H')
        df['month_year'] = df.index.to_period('M')
        df['week_year'] = df.index.to_period('W')
        return df


class ActualGenerationParser(EnergyDataTransformer):
    """
    Processes actual solar power generation reported by the ENTSOE
    Transparency Platform

    Args:
        query (str): Filters data to be processed (if needed)
        datapath (str): Base data folder
        hourly (bool): Converts all generation numbers to hourly readings
    """
    def __init__(self, query='*.csv', datapath='data', hourly=False):
        self.query = query
        self.datapath = datapath
        self.hourly = hourly
        self.COLS = ['Area', 'MTU', 'Solar  - Actual Aggregated [MW]']
    
    
    def read_actuals(self):
        list_dfs = []
        for fname in (self.in_path).glob(self.query):
            df = pd.read_csv(fname, usecols=self.COLS, encoding='utf-8')
            list_dfs.append(df)
        actuals = pd.concat(list_dfs)
        self.actuals = actuals
        
        
    def clean_actuals(self):
        self.actuals['Area'] = self.actuals['Area'].str.split('|').str[1]
        self.actuals.columns = ['operator', 'interval', 'solar']
        self.actuals = self.actuals[self.actuals['solar']!= '-']
        
    
    def parse(self):
        self.set_filepaths('actuals')
        self.read_actuals()
        self.clean_actuals()
        self.actuals = self.parse_datetimes(self.actuals)
        if self.hourly:
            self.actuals = self.convert_to_hourly(self.actuals, 'solar')
            fname = 'actuals_hourly.csv'
        else:
            fname = 'actuals.csv'
            self.int_lengths = self.calc_intervals(self.actuals)
            with open((self.out_path/'int_lengths.yaml'), 'w') as f:
                yaml.dump(self.int_lengths, f)
        self.actuals = self.add_datetimecols(self.actuals)
        self.actuals.to_csv((self.out_path/'{}'.format(fname)), index=False)
        print('Written')


class ForecastGenerationParser(EnergyDataTransformer):
    """
    Processes forecast solar power generation reported by the ENTSOE
    Transparency Platform

    Args:
        query (str): Filters data to be processed (if needed)
        datapath (str): Base data folder
        hourly (bool): Converts all generation numbers to hourly readings
    """
    def __init__(self, query='*.csv', datapath='data', hourly=False):
        self.query = query
        self.datapath = datapath
        self.hourly = hourly
        self.COLS = ['interval','operator','Generation - Solar  [MW] Day Ahead', 
                     'Generation - Solar  [MW] Intraday', 'Generation - Solar  [MW] Current']
        
        
    def read_forecasts(self):
        list_dfs = []
        for fname in (self.in_path).glob(self.query):
            df = pd.read_csv(fname, encoding='utf-8')
            cols = df.columns[1:]
            colnames = cols.str.split('/ ').str[0] 
            colnames = [col.rstrip() for col in colnames]
            colnames.insert(0, 'interval')
            df.columns = colnames
            df['operator'] = cols.str.split('/ ').str[1][0]
            list_dfs.append(df)
        forecasts = pd.concat(list_dfs)
        self.forecasts = forecasts 
        
        
    def clean_forecasts(self):
        self.forecasts = self.forecasts[self.COLS]
        self.forecasts['operator'] = self.forecasts['operator'].str.split('|').str[1]
        self.forecasts.columns = ['interval','operator','solar_da', 'solar_id', 'solar_c']
    
    
    def parse(self):
        self.set_filepaths('forecasts')
        self.read_forecasts()
        self.clean_forecasts()
        self.forecasts = self.parse_datetimes(self.forecasts)
        self.int_lengths = self.calc_intervals(self.forecasts)
        if self.hourly:
            self.forecasts = self.convert_to_hourly(self.forecasts, 'solar_da')
            fname = 'forecasts_hourly_da.csv'
        else:
            fname = 'forecasts.csv'
        self.forecasts = self.add_datetimecols(self.forecasts)
        self.forecasts.to_csv((self.out_path/'{}'.format(fname)), index=False)
        print('Written')