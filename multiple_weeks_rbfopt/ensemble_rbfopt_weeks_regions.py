#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:32:32 2021

@author: sadrachpierre
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:55:38 2021

@author: sadrachpierre
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:59:44 2021

@author: sadrachpierre
"""
from collections import OrderedDict
import math

import dlib


REGISTERED_TEST_CLASSES = OrderedDict()


def register_test_class(cls):
    REGISTERED_TEST_CLASSES[cls.name] = cls
    return cls

from scipy.optimize import differential_evolution
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
from pandas.tseries.offsets import *


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

filename = 'finalized_model.sav'

model = pickle.load(open(filename, 'rb'))

train_data = pd.read_parquet('files/file.parquet')
train_data['TIMEPERIODENDDATE'] = pd.to_datetime(train_data['TIMEPERIODENDDATE']).apply(lambda t: t.tz_localize(None))
train_data['BASE_PRICE'] = train_data['BASE_PRICE'].astype(float)

zillow = pd.read_parquet('files/zillow.parquet')
zillow['DATE'] = pd.to_datetime(zillow['DATE']).apply(lambda t: t.tz_localize(None))
zillow['TIMEPERIODENDDATE'] = zillow['DATE'].where(
zillow['DATE'] == ((zillow['DATE'] + Week(weekday=6)) - Week()),
zillow['DATE'] + Week(weekday=6))
zillow = zillow[zillow.REGION_TYPE == 'County'][['REGION_NAME', 'TIMEPERIODENDDATE', 'ZILLOW_HVI', 'ZILLOW_MEAN_DAYS', 'ZILLOW_LISTING_PRICE',
                                                 'ZILLOW_INVENTORY', 'ZILLOW_ORI']]
zillow = zillow.rename({'REGION_NAME': 'COUNTY'}, axis=1)
zillow = zillow.groupby(['TIMEPERIODENDDATE', 'COUNTY'])[['ZILLOW_HVI','ZILLOW_MEAN_DAYS', 'ZILLOW_LISTING_PRICE', 'ZILLOW_INVENTORY', 'ZILLOW_ORI']].mean()
train_data = pd.merge(train_data, zillow, on=['TIMEPERIODENDDATE', 'COUNTY'],
                           how="left")

unemployment = pd.read_parquet('files/unemployment.parquet')
print(unemployment.head())
unemployment = unemployment[['INITIAL_CLAIMS', 'CONTINUED_CLAIMS', 'COVERED_EMPLOYMENT', 'INSURED_UNEMPLOYMENT_RATE',
                             'WEEK','MONTH' ,'YEAR' ,'STATE']]
unemployment = unemployment.astype({'INITIAL_CLAIMS': 'float64'})
unemployment = unemployment.groupby(['WEEK','MONTH' ,'YEAR' ,'STATE'])[['INITIAL_CLAIMS', 'CONTINUED_CLAIMS', 'COVERED_EMPLOYMENT', 'INSURED_UNEMPLOYMENT_RATE']].mean()
train_data = pd.merge(train_data, unemployment, on=['WEEK','MONTH' ,'YEAR' ,'STATE'],
                           how="left")


NYTcovid = pd.read_parquet('files/NYTcovid.parquet')
NYTcovid['DATE'] = pd.to_datetime(NYTcovid['DATE']).apply(lambda t: t.tz_localize(None))
NYTcovid['TIMEPERIODENDDATE'] =NYTcovid['DATE'].where(NYTcovid['DATE'] == ((NYTcovid['DATE'] + Week(weekday=6)) - Week()),
                                     NYTcovid['DATE'] + Week(weekday=6))
NYTcovid.to_csv("covid.csv", index=False)
NYTcovid = NYTcovid.fillna("0")
NYTcovid = NYTcovid.rename({'REGION_NAME': 'COUNTY'}, axis=1)
NYTcovid = NYTcovid.astype({'C19_NYCASES': 'int32'})
NYTcovid = NYTcovid.astype({'C19_DEATHS': 'int32'})
NYTcovid = NYTcovid.groupby(['TIMEPERIODENDDATE','COUNTY', 'STATE'])[['C19_NYCASES', 'C19_DEATHS']].sum()
train_data = pd.merge(train_data, NYTcovid, on=['TIMEPERIODENDDATE', 'COUNTY', 'STATE'], how = "left")
train_data = train_data.fillna(0)


print(unemployment.head())

cols = ['BASE_PRICE', 'DISCOUNT_PERC', 'AVGPCTACV', 'AVGPCTACVANYDISPLAY', 'AVGPCTACVANYFEATURE', 'AVGPCTACVFEATUREANDDISPLAY', 
        'AVGPCTACVTPR', 'YEAR', 'C19_NYCASES', 'C19_DEATHS', 'INITIAL_CLAIMS','CONTINUED_CLAIMS', 'COVERED_EMPLOYMENT',
        'INSURED_UNEMPLOYMENT_RATE', 'ZILLOW_HVI', 'ZILLOW_MEAN_DAYS', 'ZILLOW_LISTING_PRICE', 'ZILLOW_INVENTORY', 'ZILLOW_ORI']

cols = ['TIMEPERIODENDDATE', 'UPC', 'BASE_PRICE', 'DISCOUNT_PERC', 'AVGPCTACV',
       'AVGPCTACVANYDISPLAY', 'AVGPCTACVANYFEATURE',
       'AVGPCTACVFEATUREANDDISPLAY', 'AVGPCTACVTPR', 'WEEK', 'MONTH', 'YEAR',
       'COUNTY', 'STATE', 'REGION', 'C19_NYCASES', 'C19_DEATHS',
       'INITIAL_CLAIMS', 'ZILLOW_HVI']

max_sppd = train_data['SPPD'].max()

week_list = list(np.arange(1,25))
geo_df = pd.read_parquet('geo_mapping.parquet', engine='fastparquet')
state_mapper = dict(zip(geo_df['SATE_NAME'], geo_df['REGION_NAME']))
county_mapper = dict(zip(geo_df['COUNTY_COUNTY_EQUIVALENT'], geo_df['SATE_NAME']))
county_list = list(set(geo_df['COUNTY_COUNTY_EQUIVALENT']))[10:19]






#['00-16300-16911', '00-48500-02119', '07-12797-15059']
upc_list = list(set(train_data['UPC']))[:10]
def model_func(X1, X2, X3, X4, X5, X6, X7, X8, X9, week, county, state, region):
    TIMEPERIODENDDATE = 0
    UPC = '00-16300-16911'#
    geo_time = [TIMEPERIODENDDATE, UPC]    
    X = [X1, X2, X3, X4, X5, X6, X7, X8, X9]
    X = np.append(geo_time, X)
    
    INITIAL_CLAIMS = train_data['INITIAL_CLAIMS'].mean()    
    ZILLOW_HVI = train_data['ZILLOW_HVI'].mean()
    BASE_PRICE = train_data['BASE_PRICE'].mean()
    ext_cols = [INITIAL_CLAIMS, ZILLOW_HVI]
    
    WEEK = week
    MONTH = 1 + int(week/4)
    YEAR = 2022
    COUNTY = county
    STATE = state
    REGION = region
    
    X = np.insert(X, 2, BASE_PRICE)        
    X = np.insert(X, 9, WEEK)
    X = np.insert(X, 10, MONTH)
    X = np.insert(X, 11, YEAR)
    X = np.insert(X, 12, COUNTY)
    X = np.insert(X, 13, STATE)
    X = np.insert(X, 14, REGION)
    X= np.append(X, ext_cols)
    X = dict(zip(cols, X ))
    X = pd.DataFrame(data=X, index=[0])
    X = X.astype(
            {'UPC': 'category', 'BASE_PRICE': 'float64', 'DISCOUNT_PERC': 'float64', 'AVGPCTACV': 'float64',
             'AVGPCTACVANYDISPLAY': 'float64', 'AVGPCTACVANYFEATURE': 'float64', 'AVGPCTACVFEATUREANDDISPLAY': 'float64', 
             'AVGPCTACVTPR': 'float64', 'WEEK': 'int64'})
   
    
    X = [X.to_dict()]
    X_new = dict()
    for key, value in X[0].items():
        X_new[key] = value[0]
        
    X = []    
    for key, value in X_new.items():
        X.append(value)
        
    #print("HERE: ", X)
    X = [X]
    y_pred = model.predict(X)
    y_pred = y_pred.reshape(1, -1)
    #model_func.counter += 1
    #print(model_func.counter)
    return y_pred.ravel()
@register_test_class
class ensemble:
    spins_bounds = [(1, 2.5), (0, 0.8), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (-0.30, 0.30), (-0.30, 0.30), (-0.30, 0.30)]

    name = "Ensemble"
    dim = 9
    lbounds = [0, 0, 0, 0, 0, 0, -0.3, -0.3, -0.3]
    ubounds = [0.8, 100, 100, 100, 100, 100, 0.3, 0.3, 0.3]
    
    optima = [(max_sppd, max_sppd), (max_sppd, max_sppd), (max_sppd, max_sppd), (max_sppd, max_sppd), (max_sppd, max_sppd), (max_sppd, max_sppd), 
              (max_sppd, max_sppd), (max_sppd, max_sppd), (max_sppd, max_sppd), (max_sppd, max_sppd)]
    fmin = -1.

    @staticmethod
    def evaluate(X):
        X1, X2, X3, X4, X5, X6, X7, X8, X9 = X
        results = []
        for week in week_list:
            for county in county_list:
                state  = county_mapper[county]
                region = state_mapper[state]
                results.append(model_func(X1, X2, X3, X4, X5, X6, X7, X8, X9, week, county, state, region))
        return np.sum(results)
        
    
    
    
def main():
    import numpy as np
    import os

    import rbfopt

    def dist(x, y):
        return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))

    solver_path = os.path.join('./Bonmin-1.8.7/build/bin')
    ipopt_file = os.path.join(solver_path, 'ipopt')
    print('ipopt: ', ipopt_file)
    if not os.path.isfile(ipopt_file):
        print('ipopt not exists!')
    bonmin_file = os.path.join(solver_path, 'bonmin')
    print('bonmin: ', bonmin_file)
    if not os.path.isfile(bonmin_file):
        print('bonmin not exists!')


    num_runs = 1
    max_fun_calls = 10
    print('num_runs =', num_runs)
    print('max_fun_calls =', max_fun_calls)
    print('Running against test functions')
    print('')
    for name, cls in REGISTERED_TEST_CLASSES.items():
        print('Function:', name)
        ndim = cls.dim
        obj_fun = cls.evaluate
        lbounds = cls.lbounds
        ubounds = cls.ubounds
        optima = cls.optima
        fmin = cls.fmin
        obj_vals = []
        fun_calls = []
        iter_counts = []
        eval_counts = []
        fast_eval_counts = []
        distances = []
        devnull = open('/dev/null', 'w')
        for k in range(num_runs):
            print(lbounds)
            print(ubounds)
            bb = rbfopt.RbfoptUserBlackBox(
                dimension=ndim,
                var_lower=np.array(lbounds, dtype=np.float),
                var_upper=np.array(ubounds, dtype=np.float),
                var_type=['R']*ndim,
                obj_funct=obj_fun)
            settings = rbfopt.RbfoptSettings(max_evaluations=max_fun_calls,
                                             minlp_solver_path=bonmin_file,
                                             nlp_solver_path=ipopt_file)
            alg = rbfopt.RbfoptAlgorithm(settings, bb)
            alg.set_output_stream(devnull)
            fval, sol, iter_count, eval_count, fast_eval_count = alg.optimize()
            obj_vals.append(fval)
            fun_calls.append(max_fun_calls)
            iter_counts.append(iter_count)
            eval_counts.append(eval_count)
            fast_eval_counts.append(fast_eval_count)
            distances.append(
                min([dist(sol, opt) for opt in optima]))
            if (k + 1) % 10 == 0:
                print('competed run', k + 1, 'of', num_runs)

        print('DISCOUNT_PERC:', sol[0])
        print('AVGPCTACV:', sol[1])
        print('AVGPCTACVANYDISPLAY:', sol[2])
        print('AVGPCTACVANYFEATURE:', sol[3])
        print('AVGPCTACVFEATUREANDDISPLAY:', sol[4])
        print('AVGPCTACVTPR:', sol[5])                

        print("Optimum objective val =", obj_vals[0])
        print('')


if __name__ == '__main__':
    import time
    start = time.time()
    print("Time elapsed on working...")
    main()
    end = time.time()
    print("Time consumed in working: ",end - start)
    
