import requests
import json
import yaml
import getpass

import pathlib, glob
import pandas as pd

import datetime
import pendulum


class WeatherPuller(object):
    """
    Pulls weather data (historical observations or forecasts)
    from the Dark Sky API
    
    Args:
        pulltype (str): Historical ('hist') or forecast ('fc') data
        coordinates (dict): Yaml file listing coordinates for each operator
        start (str): Start of time range to pull
        end(str): End of time range to pull
        datapath (str): Base data folder
        api_key (str): Dark Sky API key
    """
    def __init__(self, pulltype, start=None, end=None, coords='coordinates.yml',  datapath='data'):
        self.pulltype = pulltype
        self.coordinates = coords
        self.start = start
        self.end = end
        self.datapath = datapath
        self.api_key = getpass.getpass(prompt='api_key: ')

        
    def get_coordinates(self):
        with open(self.coordinates, 'r') as stream:
            self.coords = yaml.load(stream)
    
    
    def set_filepath(self):
        Data = pathlib.Path(self.datapath)
        path = (Data/'raw_data/weather/{}/'.format(self.pulltype))
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
    
    
    def get_timeperiods(self):
        dr = pd.date_range(self.start, self.end, freq='D')
        times = [t.strftime('%Y-%m-%dT%H:%M:%S') for t in dr]
        self.times = times    
        
        
    def gen_query_strings_hist(self):
        qss = []
        for t in self.times:
            for v in self.coords.values():
                for coord in v:
                    s = 'https://api.darksky.net/forecast/{}/{},{}?units=si'.format(self.api_key,coord,t)
                    qss.append(s)
        self.qss = qss


    def gen_query_strings_fc(self):
        qss = []
        for v in self.coords.values():
            for coord in v:
                s = 'https://api.darksky.net/forecast/{}/{}?units=si'.format(self.api_key,coord)
                qss.append(s)
        self.qss = qss

    
    def query_api(self):
        wr = []
        num_calls = len(self.qss)
        counter = 0
        for s in self.qss:
            r = requests.get(s)
            wr.append(r.json())
            counter +=1
            if counter % 50 == 0:
                print(str(round((counter/num_calls)*100))+"%")
        self.wr = wr
    
    
    def get_filename(self):
        if self.pulltype == 'hist':
            sd = self.start.split(' ')[0]
            ed = self.end.split(' ')[0]
            fn = 'wr_{}_{}.json'.format(sd, ed)
        elif self.pulltype == 'fc':
            dt = datetime.datetime.now().strftime('%Y-%m-%d')
            fn = 'wr_{}.json'.format(dt)
        self.fn = fn
    
    
    def write_json(self):
        with open((self.path/self.fn), 'w') as outfile:
            json.dump(self.wr, outfile)
        print('Written')
        
        
    def pull(self):
        self.get_coordinates()
        self.set_filepath()
        if self.pulltype == 'hist':
            self.get_timeperiods()
            self.gen_query_strings_hist()
        elif self.pulltype == 'fc': 
            self.gen_query_strings_fc()
        self.query_api()
        self.get_filename()
        self.write_json()