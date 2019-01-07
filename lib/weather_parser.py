"""
weather_parser.py:
    Parses and cleans json weather data from the Dark Sky API

    WeatherRecordParser (cls): Parses an individual weather record
    WeatherParser (cls): Wrapper for WeatherRecordParser; parses multiple records
    WeatherCleaner (cls): Cleans and transforms parsed weather data
"""


import json
import yaml
import pendulum
from pandas.io.json import json_normalize

import pathlib, glob
import pandas as pd


class WeatherRecordParser(object):
    """
    Parses an individual json weather record from the Dark Sky API
    
    Args:
        wrecord (json): Record to parse
        rectype (str): 'hist' or 'fc'
        coords (dict): Yaml file listing coordinates for each operator
        tz (str): Timezone of weather record
    """
    def __init__(self, wrecord, rectype, coords):
        self.wrecord = wrecord
        self.rectype = rectype
        self.coords = coords
        self.tz = wrecord['timezone']
        
        
    def get_operator(self):
        cd = '{},{}'.format(self.wrecord['latitude'],self.wrecord['longitude'])
        operator = [k for k,v in self.coords.items() if cd in v][0]
        coord = [v.index(cd)+1 for k,v in self.coords.items() if cd in v][0]
        self.operator, self.coord = operator, coord
    
    
    def get_local_time(self, unixtime, tz):
        utc_time = pendulum.from_timestamp(unixtime)
        local_time = utc_time.in_timezone(tz).strftime('%Y-%m-%d %H:%M:%S')
        return local_time  
    
    
    def get_pulltime(self):
        unixtime = self.wrecord['currently']['time']
        self.pulltime = self.get_local_time(unixtime, self.tz)
        
    
    def get_sunriseset_time(self):
        sunrise_uxt = self.wrecord['daily']['data'][0]['sunriseTime']
        sunset_uxt = self.wrecord['daily']['data'][0]['sunsetTime']
        self.sunrise = self.get_local_time(sunrise_uxt, self.tz)
        self.sunset = self.get_local_time(sunset_uxt, self.tz)
        
        
    def add_local_time(self, interval):
        data_block = self.wrecord[interval]['data']
        for i in range(len(data_block)):
            unixtime = data_block[i]['time']
            local_time = self.get_local_time(unixtime, self.tz)
            data_block[i]['local_time'] = local_time
        
        
    def get_nearest_station(self):
        try:
            near_st = self.wrecord['flags']['nearest-station']
        except KeyError:
            near_st = 10.0
        self.near_st = near_st  
        
    
    def add_attributes(self, df):
        df['pulltime'] = self.pulltime
        df['operator'] = self.operator
        df['coord'] = self.coord
        df['near_st'] = self.near_st
        return df
         
    
    def normalize_daily_data(self):
        if self.rectype == 'fc':
            drec = self.wrecord['daily']['data'][1]
        else:
            drec = self.wrecord['daily']['data']
        daily = json_normalize(drec)
        daily = self.add_attributes(daily)
        daily['sunrise'] = self.sunrise
        daily['sunset'] = self.sunset
        return daily
        
        
    def normalize_hourly_data(self):
        if self.rectype == 'fc':
            hrec = self.wrecord['hourly']['data'][8:32]
        else:
            hrec = self.wrecord['hourly']['data']
        hourly = json_normalize(hrec)
        hourly = self.add_attributes(hourly)
        return hourly
        
        
    def process_wrecord(self):
        self.get_operator()
        self.get_pulltime()
        self.get_sunriseset_time()
        self.get_nearest_station()
        self.add_local_time('daily')
        self.add_local_time('hourly')
        daily = self.normalize_daily_data()
        hourly = self.normalize_hourly_data()
        return daily, hourly



class WeatherParser(object):
    """
    Parses multiple json weather records from the Dark Sky API; wrapper for 
    WeatherRecordParser
    
    Args:
        query (str): Filters data to be processed (if needed)
        rectype (str): 'hist' or 'fc'
        coords (dict): Yaml file listing coordinates for each operator
        datapath (str): Base data folder
    """
    def __init__(self, rectype, query='*.json', coords='coordinates.yml', datapath='data'):
        self.rectype = rectype
        self.query = query
        self.coordinates = coords
        self.datapath = datapath
        
        
    def get_coordinates(self):
        with open(self.coordinates, 'r') as stream:
            self.coords = yaml.load(stream)
        
        
    def get_filepath(self):
        Data = pathlib.Path(self.datapath)
        path = (Data/'raw_data/weather/{}/'.format(self.rectype))
        self.path = path

        
    def read_all_files(self):
        wrecords = []
        for fname in (self.path).glob(self.query):
            with open(fname) as f:
                rec = json.load(f)
                wrecords.append(rec)
        return wrecords
    
    
    def compile_wdata(self):
        self.get_coordinates()
        self.get_filepath()
        wrecords = self.read_all_files()
        dwdata = []
        hwdata = []
        counter = 0
        for f in wrecords:
            for rec in f:
                wrp = WeatherRecordParser(rec, self.rectype, self.coords)
                dailyrec, hourlyrec = wrp.process_wrecord()
                dwdata.append(dailyrec), hwdata.append(hourlyrec)
                counter +=1
                if counter % 10000==0:
                    print('Parsed {} records'.format(counter))
        daily_weather = pd.concat(dwdata, sort=True)
        hourly_weather = pd.concat(hwdata, sort=True)
        self.daily_weather = daily_weather
        self.hourly_weather = hourly_weather
        print('Processed')



class WeatherCleaner(object):
    """
    Cleans and transforms parsed weather data
    
    Args:
        rectype (str): 'hist' or 'fc'
        weatherdata (dataframe): weather data to be processed
        interval (str): 'daily' or 'hourly'
        datapath (str): Base data folder
    """
    def __init__(self, rectype, weatherdata, interval, datapath='data'):
        self.rectype = rectype
        self.weatherdata = weatherdata
        self.interval = interval
        self.datapath = datapath
        self.REMCOLS = 'error|ozone|summary|gust|intensitymax|sunriseTime|sunsetTime'
        
        
    def remove_cols(self):
        df = self.weatherdata
        cols_to_rmv = df.columns.str.contains(r'{}'.format(self.REMCOLS), case=False)
        df = df[df.columns[~cols_to_rmv]].copy()
        self.weatherdata = df
    
    
    def extract_sunriseset_time(self):
        df = self.weatherdata
        for c in ['sunrise', 'sunset']:
            df[c] = df[c].str.split(' ').str[1]
            df[c] = df[c].replace(to_replace=r':', value='', regex=True)
            df[c] = df[c].astype(int)
        self.weatherdata = df

        
    def address_windbear_nulls(self):
        df = self.weatherdata
        df['month'] = pd.to_datetime(df['local_time']).dt.month
        df['windBearing'] = df['windBearing'].fillna(df.groupby(['operator','coord','month'])['windBearing'].transform('median'))
        self.weatherdata = df
        
        
    def interpolate_precip(self, df, col):
        df['cc_bin'] = pd.cut(df['cloudCover'],10)
        for i in [['operator','coord','month','icon','cc_bin'],
                  ['operator','coord','icon','cc_bin'],
                  ['operator','coord','icon','month'],
                  ['operator','coord','icon'],
                  ['operator','coord']]:
            df[col] = df[col].fillna(df.groupby(i)[col].transform('median'))
        return df

        
    def address_precip_nulls(self):
        df = self.weatherdata
        df['precipAccumulation'] = df['precipAccumulation'].fillna(0)
        df.loc[df['precipIntensity'] == 0, ['precipType']] = 'None'
        df = self.interpolate_precip(df, 'precipProbability')
        df = self.interpolate_precip(df, 'precipIntensity')
        df = self.interpolate_precip(df, 'visibility')
        df.drop(['month', 'cc_bin'], axis=1, inplace=True)
        self.weatherdata = df
        
        
    def align(self):
        df = self.weatherdata
        cs = []
        for i in range(1,6):
            ss = df[df['coord']==i]
            ss.columns = [str(i) + '_' + str(col) for col in ss.columns]
            cs.append(ss)
        df_shaped = pd.concat([cs[0],cs[1],cs[2],cs[3],cs[4]],axis=1)
        self.weatherdata = df_shaped
        
        
    def rmv_repeat_cols(self):
        df = self.weatherdata
        df['operator'] = df['1_operator']
        df['time'] = df['1_time']
        df['localtime'] = df['1_local_time']
        rep_cols = df.columns.str.contains(r'_operator|_time|_coord', case=False)
        df_shaped = df[df.columns[~rep_cols]].copy()
        self.weatherdata = df_shaped
        
        
    def set_filepath(self):
        Data = pathlib.Path(self.datapath)
        path = (Data/'processed_data/')
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        
        
    def write(self):
        self.set_filepath()
        self.weatherdata.to_csv((self.path/'{}_weather_{}.csv'.format(self.interval, self.rectype)), index=False)
        print('Written')
        
        
    def clean(self):
        self.remove_cols()
        if self.interval == 'daily':
            self.extract_sunriseset_time()
        self.address_windbear_nulls()
        self.address_precip_nulls()
        self.align()
        self.rmv_repeat_cols()
        self.write()