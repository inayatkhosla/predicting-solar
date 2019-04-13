"""
modelers.py:
    Trains, validates, and tests Random Forests, XGBoost, KNN,
    and Neural Nets across a range of operators, time periods, 
    and hyperparameters. The network can be trained using fastai
    (for those new to deep learning) or pytorch (for those with
    more experience)

    PredictionGenerator (cls): Wrapper - generates predictions across a 
    range of operators and time periods
    GenerationModeler (cls): Generates predictions for a given operator 
    and time period
    ModelPrepper (cls): Prepares data for modeling
    KNNModeler (cls): Trains and predicts using K Nearest Neighbors
    TorchNNModeler (cls): Trains and predicts using a neural network with 
    entity embeddings - pytorch
    NNModeler (cls): Trains and predicts using a neural network with 
    entity embeddings - fastai

"""

import pathlib
import pandas as pd
import numpy as np

import xgboost as xgb
import nmslib # a much faster implementation of KNN than sklearn's
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fastai.structured import *  
from fastai.column_data import * 

import torch
import torch.optim as optim
import torch.nn as nn
from functools import partial

from lib import torch_tabular as t
from lib import cyclic_lr as c


class PredictionGenerator(object):
    """
    Wrapper - generates predictions across a range of
    operators and time periods

    Args:
        data (df): Base model data
        operators (list): List of operators to generate predictions for
        features (dict): Features to include by type
        params (dict): Hyperparameters across model types
        train_start (str): Train period start
        prediction_periods (list): Periods for which to generate predictions
        period_length (int): Prediction period length in days
        val (bool): Whether predictions are for the validation or test set
        pytorch (bool): Whether to use fastai or pytorch
    """
    def __init__(self, data, operators, features, params, train_start, 
                 prediction_periods, period_length, val=True, pytorch=True):
        self.data = data
        self.operators = operators
        self.features = features
        self.params = params
        self.train_start = train_start
        self.prediction_periods = prediction_periods
        self.period_length = period_length
        self.val = val
        self.pytorch = pytorch
        
        
    def generate_operator_predictions(self, operator):
        p = []
        for per in self.prediction_periods:
            print('predicting {}'.format(per))
            gm = GenerationModeler(self.data, operator, self.train_start, per, 
                 self.period_length, self.params, self.features, self.pytorch)
            gm.run()
            p.append(gm.preds)
        preds = pd.concat(p, sort=True)
        return preds
        
        
    def generate_overall_predictions(self):
        all_preds = []
        for o in self.operators:
            print(o)
            preds = self.generate_operator_predictions(o)
            preds['operator'] = o
            all_preds.append(preds)
        self.predictions = pd.concat(all_preds, sort=True)
            
    
    def write(self):
        print('writing')
        out_path = pathlib.Path('results')
        out_path.mkdir(parents=True, exist_ok=True)
        if sel.val:
            fname = 'valpreds.csv'
        else:
            fname = 'testpreds.csv'
        self.predictions.to_csv((out_path/{}), index=False).format(fname)
    
    
    def run(self):
        self.generate_overall_predictions()
        #self.write()



class GenerationModeler(object):
    """
    Generates predictions for a given operator and time period

    Args:
        data (df): Base model data
        operator (str): Operator to generate predictions for
        train_start (str): Train period start
        val_start (str): Prediction period start
        val_length (int): Prediction period length in days
        params (dict): Hyperparameters across model types
        pytorch (bool): Whether to use fastai or pytorch
    """
    def __init__(self, all_data, operator, train_start, val_start, val_length, params, features, pytorch):
        self.all_data = all_data
        self.operator = operator
        self.train_start = train_start
        self.val_start = val_start
        self.val_length = val_length
        self.params = params
        self.features = features
        self.pytorch = pytorch
        self.DEP = 'solar'
        
        
        
    def get_data(self):
        mp = ModelPrepper(self.all_data, self.operator, self.features)
        mp.run()
        self.data = mp.md
        
    
    def train_val_split(self):
        self.val_start = pd.to_datetime(self.val_start)
        self.val_end = self.val_start + pd.Timedelta(days=self.val_length)
        self.train = self.data.loc[(self.data.index >= self.train_start) & (self.data.index < self.val_start)]
        self.val = self.data.loc[(self.data.index >= self.val_start) & (self.data.index < self.val_end)]
        
    
    def xy_split(self):
        dep = self.DEP
        features = self.data.drop(self.DEP, axis=1).columns
        self.X_train, self.y_train = self.train[features].values, self.train[dep].values
        self.X_val, self.y_val = self.val[features].values, self.val[dep].values
        
        
    def fit_predict_rf(self):
        print('Running Random Forests')
        params = self.params['rf']
        mdl = RandomForestRegressor(n_jobs=-1, random_state=0, **params)
        mdl.fit(self.X_train, self.y_train)
        self.rf_val = mdl.predict(self.X_val)
        self.rf_mdl = mdl
        
    
    def fit_predict_xgb(self):
        print('Running XGBoost')
        params = self.params['xgb']
        bst = xgb.XGBRegressor(**params)
        bst.fit(self.X_train,self.y_train)
        self.xgb_val = bst.predict(self.X_val)
        self.xgb_mdl = bst
        
    
    def fit_predict_knn(self):
        print('Running KNN')
        knn =  KNNModeler(self.X_train, self.X_val, self.y_train, self.params)
        knn.run()
        self.knn_val  = knn.knn_val
        self.knn_mdl  = knn.knn_mdl
        
    
    def fit_predict_nn(self):
        print('Running NN')
        nmp = ModelPrepper(self.all_data, self.operator, self.features, nn=True)
        nmp.run()
        nndata = nmp.md
        nn = NNModeler(self.operator, nndata, self.params, nmp.catvars, self.train_start, self.val_start, self.val_end)
        self.nn_val = nn.fit_predict()
        self.nn_mdl = nn.mdl


    def fit_predict_torch_nn(self):
        print('Running NN')
        tnn = TorchNN(md, operator, params, catvars, train_start, val_start, val_length, dep, log_y=True, bs=128)
        self.nn_val = tnn.fit_predict()
        self.nn_mdl = tnn.model
        
        
    # refactor    
    def combine(self):
        print('Combining')
        preds = pd.DataFrame(list(zip(self.val.reset_index()['int_start'], self.y_val, 
                                      self.rf_val, self.xgb_val, self.knn_val, self.nn_val)))
        preds.columns = ['int_start','solar','rf', 'xgb', 'knn', 'nn']
        preds['nn'] = preds['nn'].str[0]
        self.preds = preds
    
    
    def run(self):
        self.get_data()
        self.train_val_split()
        self.xy_split()
        self.fit_predict_rf()
        self.fit_predict_xgb()
        self.fit_predict_knn()
        if self.pytorch:
            self.fit_predict_torch_nn()
        else:
            self.fit_predict_nn()
        self.combine()


class ModelPrepper(object):
    """
    Prepares the data for modeling - filters for features of interest,
    and generates categorical and dummy variables

    Args:
        data (df): Base model data
        operator (str): Operator to generate predictions for
        features (dict): Features to include by type
        nn (bool): Whether the data being generated is for a neural net
    """
    def __init__(self, data, operator, features, nn=False):
        self.data = data
        self.operator = operator
        self.features = features
        self.nn = nn
        self.DEP = 'solar'

        
    def get_operator_data(self):
        md = self.data
        md = md.loc[md['operator'] == self.operator]
        md[self.DEP] = pd.to_numeric(md[self.DEP], errors='coerce')
        md = md.loc[md[self.DEP] !=0]
        #md = md.loc[(md['hour'] > 7) & (md['hour'] < 20)]
        self.md = md.set_index(pd.DatetimeIndex(md['int_start']))
        
    
    
    def filter_features(self):
        features = self.features
        cols = self.md.columns
        dcols = cols[cols.str.contains('d_')]
        hcols = cols[cols.str.startswith('h_')]

        base = list(cols[cols.str.contains(r'{}'.format(features['base']), case=False)])
        trailing = list(cols[cols.str.contains(r'{}'.format(features['trailing']), case=False)])
        capacity = list(cols[cols.str.contains(r'{}'.format(features['capacity']), case=False)])
        daily = list(dcols[dcols.str.contains(r'{}'.format(features['daily']), case=False)])
        hourly = list(hcols[hcols.str.contains(r'{}'.format(features['hourly']), case=False)])
        
        self.feature_list = base + trailing + capacity + daily + hourly
        self.md = self.md[self.feature_list + [self.DEP]]
        
        
        
    def get_catvars(self):
        cols = self.md.columns
        self.catvars = list(cols[cols.str.contains(r'{}'.format(self.features['base'] + '|icon|preciptype|uvindex'), case=False)])
        
    
    
    def generate_dummies(self, df, obs_type):
        for l in ['d','h']:
            for i in range(1,6):
                for c in df.columns[df.columns.str.contains('{}_{}_{}'.format(l,i,obs_type))]:
                    df = pd.concat([df, pd.get_dummies(df[c], prefix='{}_{}_{}'.format(l,i, obs_type))], axis=1)
                    df.drop(c, axis=1, inplace=True)
        return df    
    
    
    def run(self):
        self.get_operator_data()
        self.filter_features()
        if self.nn:
            self.get_catvars()
        else:
            for i in ['icon', 'precipType']:
                self.md = self.generate_dummies(self.md, i)



class KNNModeler(GenerationModeler):
    """
    Trains and predicts using K Nearest Neighbors

    Args:
        X_train (array): Feature data to train on
        X_val (array): Feature data to predict
        y_train (array): Targets to train on
        params (dict): Hyperparameters across model types
    """
    def __init__(self, X_train, X_val, y_train, params):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.params = params
    
    
    def scale(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
    
    
    def create_index(self):
        index = nmslib.init(space=self.params['knn']['space'])
        index.addDataPointBatch(self.X_train)
        index.createIndex(print_progress=True)
        self.index = index
    
    
    def get_preds(self, val):
        idxs, dists = zip(*self.index.knnQueryBatch(val, k=self.params['knn']['k'], num_threads=4))
        y_pred = [np.median([self.y_train[i] for i in idx]) for idx in idxs]
        self.ranges = list(zip([np.max([self.y_train[i] for i in idx]) for idx in idxs],[np.min([self.y_train[i] for i in idx]) for idx in idxs]))
        return y_pred
    
    
    def run(self):
        self.scale()
        self.create_index()
        self.knn_val = self.get_preds(self.X_val)
        self.knn_mdl = self.index      


class TorchNN(object):
    """
    Trains and predicts using a neural network with entity embeddings - pytorch

    Args:
        md (df): Prepared model data for the operator
        operator (str): Operator to generate predictions for
        params (dict): NN hyperparameters
        catvars (list): List of categorical variables
        train_start (str): Train period start
        val_start (str): Prediction period start
        val_length (int): Prediction period length in days
        dep (str): Target variable
        log_y (bool): Whether to convert target variable to log
        bs (int): Data loader batch size
    """
    def __init__(self, md, operator, params, catvars, train_start, val_start, val_length, dep, log_y=True, bs=128):
        self.md = md
        self.operator = operator
        self.params = params['torch_nn']
        self.catvars = catvars
        self.train_start = train_start
        self.val_start = val_start
        self.val_length = val_length
        self.dep = dep
        self.log_y = log_y
        self.bs = bs
        self.OUT_SZ = 1
        self.CRITERION = nn.MSELoss()
        
    
    def preprocess(self):
        tdp = t.TorchTabularDataPrepper(self.md, self.catvars, self.train_start, self.val_start, 
                                        self.val_length, self.dep, self.log_y, self.bs)
        tdp.run()
        self.trainloader = tdp.trainloader
        self.valloader = tdp.valloader
        self.emb_szs = tdp.emb_szs
        self.continvars = tdp.continvars
        self.y_range = tdp.y_range
        self.valid = tdp.val
        
        
    def instantiate_model(self):
        self.model = t.TabularModel(emb_szs=self.emb_szs, 
                                    n_cont=len(self.continvars),
                                    emb_drop=self.params['emb_drop'], 
                                    out_sz=self.OUT_SZ, 
                                    layers=self.params['szs'], 
                                    ps=self.params['drops'], 
                                    y_range=self.y_range)
        
    
    def get_training_params(self):
        self.n_epochs = self.params['n_epochs']
        max_lr = self.params['lr'][self.operator]
        base_lr = max_lr/10
        AdamW = partial(optim.Adam, betas=(0.9,0.99))
        self.optimizer = AdamW(self.model.parameters(), lr=max_lr, weight_decay=self.params['wd'])
        self.scheduler = c.CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr, step_size=self.params['step_size'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
    def set_modelpath(self):
        model_path = pathlib.Path('models')
        model_name = self.operator + '.pt'
        self.model_path = model_path/model_name
    
    
    def fit(self):
        valid_loss_min = np.Inf
        
        for epoch in range(self.n_epochs):
            train_loss = 0.0
            valid_loss = 0.0
            
            ## TRAIN ##
            self.model.train()
            for x_cat, x_cont, y in self.trainloader:
                x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x_cat, x_cont)
                loss = self.CRITERION(pred, y)
                loss.backward()
                self.scheduler.batch_step()
                self.optimizer.step()
                train_loss += loss.item()
            
        
            ## VALIDATE ##
            self.model.eval()
            with torch.no_grad():
                for x_cat, x_cont, y in self.valloader:
                    x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device)
                    pred = self.model(x_cat, x_cont)
                    loss = self.CRITERION(pred, y)
                    valid_loss += loss.item()
        
        
            train_loss = train_loss/len(self.trainloader)
            valid_loss = valid_loss/len(self.valloader)

            print(f'Epoch {epoch + 1}: trn_loss: {train_loss:.2f} | val_loss: {valid_loss:.2f}')

            if valid_loss < valid_loss_min:
                torch.save(self.model.state_dict(), self.model_path)

    
    def predict(self):
        self.model.load_state_dict(torch.load(self.model_path))
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for x_cat, x_cont, y in self.valloader:
                x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device)
                pred = self.model(x_cat, x_cont)
                pred = pred.contiguous().view(-1)
                outputs.append(pred)
        preds = np.concatenate(outputs)
        return preds
    
    
    def fit_predict(self):
        self.preprocess()
        self.instantiate_model()
        self.get_training_params()
        self.set_modelpath()
        self.fit()
        if self.log_y:
            predictions = np.exp(self.predict())
        else:
            predictions = self.predict()
        return predictions


    class NNModeler(object):
    """
    Trains and predicts using a neural network with entity embeddings - fastai

    Args:
        operator (str): Operator to generate predictions for
        data (df): Prepared model data for the operator
        rparams (dict): NN hyperparameters
        train_start (str): Train period start
        val_start (str): Prediction period start
        val_end (str): Prediction period end
        test_end (str): Test period end
    """
    def __init__(self, operator, data, params, catvars, train_start, val_start, val_end, test_end='2018-10-05'):
        self.operator = operator
        self.data = data
        self.rparams = params['nn']
        self.catvars = catvars
        self.train_start = train_start
        self.val_start = val_start
        self.val_end = val_end
        self.test_end = test_end
        self.MODEL_PATH = 'models'
        self.DEP = 'solar'
        self.prep()

    
    def train_test_split(self):
        data = self.data
        self.train = data.loc[(data.index >= self.train_start) & (data.index <= self.val_end)]
        self.test = data.loc[(data.index > self.val_end) & (data.index <= self.test_end)]

    
    def generate_embeddings(self):
        for v in self.catvars:
            self.train[v] = self.train[v].astype('category').cat.as_ordered()
        apply_cats(self.test, self.train)
        cat_sz = [(c, len(self.train[c].cat.categories)+1) for c in self.catvars]
        ## Rule of thumb for embedding sizes
        self.emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
        
    
    def adjust_cont_datatypes(self):
        self.continvars = list(set(self.train.columns) - set(self.catvars))
        for v in self.continvars:
            self.train[v] = self.train[v].astype('float32')
            self.test[v]  = self.test[v].astype('float32')
    
    
    def setup_modeldataobject(self):
        ## Fastai functionality - preps data (scales, maps, sructures, sets up data loaders)
        df, y, nas, mapper = proc_df(self.train, self.DEP, do_scale=True)
        df_test, _, nas, mapper = proc_df(self.test, self.DEP, do_scale=True, mapper=mapper, na_dict=nas)
        df.index = pd.to_datetime(df.index)
        self.yl = np.log(y)
        val_idx = np.flatnonzero((df.index >= self.val_start) & (df.index <= self.val_end))
        self.md = ColumnarModelData.from_data_frame(self.MODEL_PATH, val_idx, df, self.yl, 
                                                    cat_flds=self.catvars, bs=128, test_df=df_test)
        
    def get_yrange(self):
        # Limit y range
        max_log_y = np.max(self.yl)
        self.y_range = (0, max_log_y*1.05)
        

    def setup_learner(self):
        self.m = self.md.get_learner(self.emb_szs, len(self.continvars)-1,
                                     self.rparams['emb_drop'], 1, self.rparams['szs'], 
                                     self.rparams['drops'], y_range=self.y_range, 
                                     use_bn=self.rparams['use_bn'])
        
        
    def find_optimal_lr(self):
        self.m.lr_find()
        self.m.sched.plot()
        
        
    def inv_y(self, a): 
        return np.exp(a)

    
    def mdpe(self, targ, y_pred): 
        targ = self.inv_y(targ)
        pred = self.inv_y(y_pred)
        atarg, apred = np.array(targ), np.array(pred)
        return np.median(np.abs((atarg - apred) / atarg)) * 100
        
    
    def fit_learner(self):
        # this earlier version of fastai doesn't support early stopping
        # it is implemented using pytorch below
        lr = self.rparams['lr'][self.operator]
        self.m.fit(lr, 5, metrics=[self.mdpe], cycle_len=1)
        #SGD with restarts
        self.m.fit(lr, 2, metrics=[self.mdpe], cycle_len=3)
        self.m.fit(lr, 3, cycle_len=1, cycle_mult=2, metrics=[self.mdpe])
        self.mdl = self.m
        
        
    def fit_predict(self):
        self.setup_modeldataobject()
        self.get_yrange()
        self.setup_learner()
        self.fit_learner()
        predictions = np.exp(self.mdl.predict())
        return predictions
    

    def prep(self):
        self.train_test_split()
        self.generate_embeddings()
        self.adjust_cont_datatypes()