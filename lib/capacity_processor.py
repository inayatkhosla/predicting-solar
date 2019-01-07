import pathlib, glob
import pandas as pd


class CapacityParser(object):
    """
    Processes installed capacity data reported by the ENTSOE
    Transparency Platform

    Args:
        query (str): Filters data to be processed (if needed)
        datapath (str): Base data folder
    """
    def __init__(self, query='*', datapath='data'):
        self.query = query
        self.datapath = datapath
    
    def set_filepaths(self):
        Data = pathlib.Path(self.datapath)
        self.in_path = (Data/'raw_data/inst_cap/')
        self.out_path = (Data/'processed_data/')
        self.out_path.mkdir(parents=True, exist_ok=True)
    
    
    def read_all_files(self):
        cap_files = []
        for fname in (self.in_path).glob(self.query):
            operator = str(fname.name).split('_')[0]
            cap_file = pd.read_csv(fname)
            cap_file['operator'] = operator
            cap_files.append(cap_file)
        cap_data = pd.concat(cap_files)
        self.cap_data = cap_data
        
        
    def reshape_data(self):
        df = self.cap_data
        n_ops = df['operator'].nunique()
        df = df[df['Production Type']=='Solar']
        dfs = df.melt(id_vars=['operator'], var_name='year', value_name='solcap')[n_ops:]
        dfs['year'] = dfs['year'].str.split(' ').str[0]
        self.cap_data = dfs
        
    def correct_operator_names(self):
        op_map = {'DE50Hertz':'DE(50Hertz)',
                  'DEAmprion':'DE(Amprion)',
                  'DETenneTGER':'DE(TenneT GER)',
                  'DETransnetBW':'DE(TransnetBW)'} 
        self.cap_data['operator'] = self.cap_data['operator'].replace(op_map)

        
    def parse(self):
        self.set_filepaths()
        self.read_all_files()
        self.reshape_data()
        self.correct_operator_names()
        self.cap_data.to_csv((self.out_path/'capacity.csv'), index=False)
        print('Written')