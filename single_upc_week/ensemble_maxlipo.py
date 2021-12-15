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


import numpy as np
import pandas as pd
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
#NYTcovid.to_csv("covid.csv", index=False)
NYTcovid = NYTcovid.fillna("0")
NYTcovid = NYTcovid.rename({'REGION_NAME': 'COUNTY'}, axis=1)
NYTcovid = NYTcovid.astype({'C19_NYCASES': 'int32'})
NYTcovid = NYTcovid.astype({'C19_DEATHS': 'int32'})
NYTcovid = NYTcovid.groupby(['TIMEPERIODENDDATE','COUNTY', 'STATE'])[['C19_NYCASES', 'C19_DEATHS']].sum()
train_data = pd.merge(train_data, NYTcovid, on=['TIMEPERIODENDDATE', 'COUNTY', 'STATE'], how = "left")
train_data = train_data.fillna(0)


# train_data = train_data.astype(
#     {'BASE_PRICE': 'float64', 'DISCOUNT_PERC': 'float64', 'AVGPCTACV': 'float64',
#      'AVGPCTACVANYDISPLAY': 'float64', 'AVGPCTACVANYFEATURE': 'float64'
#         , 'AVGPCTACVFEATUREANDDISPLAY': 'float64', 'AVGPCTACVTPR': 'float64', 'WEEK': 'int64', 'INITIAL_CLAIMS':'float64'})


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
    def evaluate(X1, X2, X3, X4, X5, X6, X7, X8, X9):
        TIMEPERIODENDDATE = 0#unemployment['CONTINUED_CLAIMS'].mean()
        UPC = '00-16300-16911'#unemployment['COVERED_EMPLOYMENT'].mean()
        geo_time = [TIMEPERIODENDDATE, UPC]    
        X = [X1, X2, X3, X4, X5, X6, X7, X8, X9]
        X = np.append(geo_time, X)
        
        INITIAL_CLAIMS = train_data['INITIAL_CLAIMS'].mean()    
        ZILLOW_HVI = train_data['ZILLOW_HVI'].mean()
        BASE_PRICE = train_data['BASE_PRICE'].mean()
        ext_cols = [INITIAL_CLAIMS, ZILLOW_HVI]
        
        WEEK = 1
        MONTH = 1
        YEAR = 2022
        COUNTY = 'Valencia County'
        STATE = 'New Mexico'
        REGION = 'West Region'
        
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
            
        print("HERE: ", X)
        X = [X]
        y_pred = model.predict(X)
        y_pred = y_pred.reshape(1, -1)
        #model_func.counter += 1
        #print(model_func.counter)
        return y_pred.ravel()
    
    
    
def main():
    import numpy as np

    def dist(x, y):
        return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))

    num_runs = 1
    max_fun_calls = 200
    print('num_runs =', num_runs)
    print('max_fun_calls =', max_fun_calls)
    print('Running against test functions')
    print('')
    for name, cls in REGISTERED_TEST_CLASSES.items():
        print('Function:', name)
        obj_fun = cls.evaluate
        lbounds = cls.lbounds
        ubounds = cls.ubounds
        optima = cls.optima
        obj_vals = []
        fun_calls = []
        distances = []
        for k in range(num_runs):
            sol, obj_val = dlib.find_max_global(obj_fun, lbounds, ubounds, max_fun_calls)
            obj_vals.append(obj_val)
            fun_calls.append(max_fun_calls)
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

        print("Optimum objective val =", obj_val)
        print('')


if __name__ == '__main__':
    import time
    start = time.time()
    print("Time elapsed on working...")
    main()
    end = time.time()
    print("Time consumed in working: ",end - start)