"""
torch_tabular.py:
    Converts mixed input tabular data into pytorch dataloaders with 
    categorical embeddings and continuous variables; defines tabular
    model architecture

    TabularDataset (cls): Converts dataframe to pytorch dataset
    TorchTabularDataPrepper (cls): Converts tabular mixed input data to pytorch 
    dataloaders with categorical embeddings and continuous variables
    TabularModel (cls): Fully connected network architecture with entity embeddings

"""
   
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


######## Data Prep 

class TabularDataset(Dataset):
    """
    Converts dataframe to pytorch dataset

    Args:
        data (df): Base model data
        catvars (list): List of categorical variables
        continvars (list): List of continuous variables
        target (str): Target variable
    """
    def __init__(self, data, catvars=None, continvars=None, target=None):
        self.n = data.shape[0]
        
        if target:
            self.y = data[target].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))
        
        self.catvars = catvars if catvars else []
        self.continvars = continvars if continvars else []
        
        if self.catvars:
            self.cat_X = data[catvars].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.n, 1))
        
        if self.continvars:
            self.cont_X = data[self.continvars].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))
            
        
    def __len__(self):
        return self.n
            
    
    def __getitem__(self, idx):
        return [self.cat_X[idx], self.cont_X[idx], self.y[idx]]



class TorchTabularDataPrepper(object):
    """
    Converts tabular mixed input data to pytorch dataloaders with categorical 
    embeddings and continuous variables

    Args:
        data (df): Base model data
        catvars (list): List of categorical variables
        continvars (list): List of continuous variables
        train_start (str): Train period start
        val_start (str): Prediction period start
        val_length (int): Prediction period length in days
        dep (str): Target variable
        log_y (bool): Whether to convert target variable to log
        bs (int): Data loader batch size
        num_workers (int): Num workers for data loader
        target (str): Target variable
    """
    def __init__(self, data, catvars, train_start, val_start, val_length, dep, 
                 log_y=True, bs=128, num_workers=1):
        self.data = data
        self.catvars = catvars
        self.train_start = train_start
        self.val_start = val_start
        self.val_length = val_length
        self.dep_o = dep
        self.log_y = log_y
        self.bs = bs
        self.num_workers = num_workers
        
    
    def get_log_y(self):
        self.data['target_log'] = np.log(self.data[self.dep_o])
        self.dep = 'target_log'
    
    
    def train_val_split(self):
        self.val_start = pd.to_datetime(self.val_start)
        self.val_end = self.val_start + pd.Timedelta(days=self.val_length)
        self.train = self.data.loc[(self.data.index >= self.train_start) & (self.data.index < self.val_start)]
        self.val = self.data.loc[(self.data.index >= self.val_start) & (self.data.index < self.val_end)]
        for i in self.train, self.val:
            i.reset_index(drop=True, inplace=True)
            

    ###### Categorical Feature Handling
    def assign_categories(self):
        self.categories = {}
        for n in self.catvars:
            self.train.loc[:,n] = self.train.loc[:,n].astype('category').cat.as_ordered()
            self.categories[n] = self.train[n].cat.categories
            self.val.loc[:,n] = pd.Categorical(self.val[n], categories=self.categories[n], ordered=True)
          
            
    def get_emb_szs(self):
        # Rules of thumb
        cat_sz = [(c, len(self.train[c].cat.categories)+1) for c in self.catvars]
        self.emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
        # self.emb_szs = [(c, min(600, round(1.6 * c**0.56))) for _,c in cat_sz]
    
    
    def numericalize_cats(self):
        for c in self.catvars: 
            self.train[c] = self.train[c].cat.codes + 1
            self.val[c] = self.val[c].cat.codes + 1
            
    
    def handle_catfeatures(self):
        self.assign_categories()
        self.get_emb_szs()
        self.numericalize_cats()
    
    
    ####### Continuous Feature Handling
    def get_continvars(self):
        self.continvars = list(set(self.train.columns) - set(self.catvars))
        self.continvars = [e for e in self.continvars if e not in (self.dep, self.dep_o)]
        
    
    def normalize(self):
        means, stds = {},{}
        for n in self.continvars:
            means[n], stds[n] = self.train.loc[:,n].mean(), self.train.loc[:,n].std()
            self.train.loc[:,n] = (self.train.loc[:,n] - means[n]) / (1e-7 + stds[n])
            self.val.loc[:,n] = (self.val.loc[:,n] - means[n]) / (1e-7 + stds[n])
            
            
    def handle_contfeatures(self):
        self.get_continvars()
        self.normalize()
    
    
    def set_yrange(self):
        y_min = self.data[self.dep].min()*0.9
        y_max = self.data[self.dep].max()*1.5
        self.y_range = (y_min, y_max)
        
        
    def get_dataloaders(self):
        self.trainset = TabularDataset(self.train, self.catvars, self.continvars, self.dep)
        self.valset = TabularDataset(self.val, self.catvars, self.continvars, self.dep)
        self.trainloader = DataLoader(self.trainset, batch_size=self.bs, shuffle=True, num_workers=self.num_workers)
        self.valloader = DataLoader(self.valset, batch_size=self.bs, shuffle=False, num_workers=self.num_workers)
    
    
    def run(self):
        if self.log_y:
            self.get_log_y()
        self.train_val_split()
        self.handle_catfeatures()
        self.handle_contfeatures()
        self.set_yrange()
        self.get_dataloaders()



######## Modeling

def bn_drop_lin(n_in, n_out, bn, p, actn=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization."
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


def embedding(ni,nf):
    "Create an embedding layer."
    emb = nn.Embedding(ni, nf)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad(): 
        trunc_normal_(emb.weight, std=0.01)
    return emb


class TabularModel(nn.Module):
    """
    Fully connected network architecture with entity embeddings
    See https://github.com/fastai/fastai/tree/master/fastai

    Args:
        emb_szs (list of tuples): Number of categories in, number of features
        for each categorical variable 
        n_cont (int): Number of continuous variables
        out_sz (int): Number of output categories (1 for regression)
        layers (list of ints): Hidden layers and their sizes
        ps (list of floats): Dropout for each hidden layer
        emb_drop (float): Embedding dropout
        y_range (range): Target variable output range
        use_bn (bool): Whether to use Batch Normalization
        bn_final (bool): Whether to use Batch Normalization in the final layer
    """
    
    def __init__(self, emb_szs, n_cont, out_sz, layers, ps, emb_drop, y_range=None, use_bn=True, bn_final=False):
        super().__init__()
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb, self.n_cont, self.y_range = n_emb, n_cont, y_range
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True)] * (len(sizes)-2) + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x




