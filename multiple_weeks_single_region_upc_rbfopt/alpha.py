import wqpt
import pandas as pd 
from subalphas.alpha49 import Alpha as Alpha49
from subalphas.alpha50 import Alpha as Alpha50
from sklearn.ensemble import VotingRegressor

from pandas.tseries.offsets import Week

import pickle

class Alpha:
    def __init__(self):
        # Example how to work with alpha's state:
        # self.observed_states = []
        self.Alpha49 = Alpha49 #10may 0.99 0.95
        self.Alpha50 = Alpha50 #workbench-e8cd76a2 AUROC 0.99 AP 0.97

        pd.set_option('display.max_columns', None)
        self.train_data = pd.read_parquet('files/file.parquet')#.sample(100)
        self.train_data['TIMEPERIODENDDATE'] = pd.to_datetime(self.train_data['TIMEPERIODENDDATE']).apply(lambda t: t.tz_localize(None))

        print("TRAIN_DATA ORIGIN: ")
        print(self.train_data)

        self.zillow = pd.read_parquet('files/zillow.parquet').sample(100)
        self.zillow['DATE'] = pd.to_datetime(self.zillow['DATE']).apply(lambda t: t.tz_localize(None))
        self.zillow['TIMEPERIODENDDATE'] = self.zillow['DATE'].where(
            self.zillow['DATE'] == ((self.zillow['DATE'] + Week(weekday=6)) - Week()),
            self.zillow['DATE'] + Week(weekday=6))
        self.zillow = self.zillow[self.zillow.REGION_TYPE == 'County'][['REGION_NAME', 'TIMEPERIODENDDATE', 'ZILLOW_HVI']]
        self.zillow = self.zillow.rename({'REGION_NAME': 'COUNTY'}, axis=1)
        self.zillow = self.zillow.groupby(['TIMEPERIODENDDATE', 'COUNTY'])[['ZILLOW_HVI']].mean()
        self.train_data = pd.merge(self.train_data, self.zillow, on=['TIMEPERIODENDDATE', 'COUNTY'],
                                   how="left")

        print("TRAIN_DATA AFTER MERGE ZILLOW: ")
        print(self.train_data)

        self.unemployment = pd.read_parquet('files/unemployment.parquet')
        self.unemployment = self.unemployment[['INITIAL_CLAIMS', 'WEEK','MONTH' ,'YEAR' ,'STATE']]
        self.unemployment = self.unemployment.astype({'INITIAL_CLAIMS': 'float64'})
        self.unemployment = self.unemployment.groupby(['WEEK','MONTH' ,'YEAR' ,'STATE'])[['INITIAL_CLAIMS']].mean()
        self.train_data = pd.merge(self.train_data, self.unemployment, on=['WEEK','MONTH' ,'YEAR' ,'STATE'],
                                   how="left")

        print("TRAIN_DATA AFTER MERGE UNEMPLOYMENT: ")
        print(self.train_data)

        self.NYTcovid = pd.read_parquet('files/NYTcovid.parquet')
        self.NYTcovid['DATE'] = pd.to_datetime(self.NYTcovid['DATE']).apply(lambda t: t.tz_localize(None))
        self.NYTcovid['TIMEPERIODENDDATE'] = self.NYTcovid['DATE'].where(self.NYTcovid['DATE'] == ((self.NYTcovid['DATE'] + Week(weekday=6)) - Week()),
                                             self.NYTcovid['DATE'] + Week(weekday=6))
        self.NYTcovid.to_csv("covid.csv", index=False)
        self.NYTcovid = self.NYTcovid.fillna("0")
        self.NYTcovid = self.NYTcovid.rename({'REGION_NAME': 'COUNTY'}, axis=1)
        self.NYTcovid = self.NYTcovid.astype({'C19_NYCASES': 'int32'})
        self.NYTcovid = self.NYTcovid.astype({'C19_DEATHS': 'int32'})
        self.NYTcovid = self.NYTcovid.groupby(['TIMEPERIODENDDATE','COUNTY', 'STATE'])[['C19_NYCASES', 'C19_DEATHS']].sum()
        self.train_data = pd.merge(self.train_data, self.NYTcovid, on=['TIMEPERIODENDDATE', 'COUNTY', 'STATE'], how = "left")
        self.train_data = self.train_data.fillna(0)
        print("TRAIN_DATA AFTER MERGE NYTcovid: ")
        print(self.train_data)

        self.train_data = self.train_data.astype(
            {'BASE_PRICE': 'float64', 'DISCOUNT_PERC': 'float64', 'AVGPCTACV': 'float64',
             'AVGPCTACVANYDISPLAY': 'float64', 'AVGPCTACVANYFEATURE': 'float64'
                , 'AVGPCTACVFEATUREANDDISPLAY': 'float64', 'AVGPCTACVTPR': 'float64', 'WEEK': 'int64', 'INITIAL_CLAIMS':'float64'})

        self.X_train = self.train_data.drop(columns=['SPPD'])

        self.cat_features = []
        for col, dtype in self.X_train.dtypes.to_dict().items():
            if str(dtype) in (['string', 'object']):
                self.X_train[col] = pd.Series(self.X_train[col], dtype='category')
                self.X_train[col] = pd.Series(self.X_train[col], dtype='category')

        self.y_train = self.train_data['SPPD']    
    def set_state(self, ):
        # Example:
        # self.observed_states.append([])
        pass
    
    
    def model_instance(self, alpha_in):
        self.model_in = alpha_in()
        self.model_in.fit()  
        self.model_in_model = self.model_in.model_base

        return self.model_in_model

    def fit(self):
        self.model49 = self.model_instance(self.Alpha49)
        self.model50 = self.model_instance(self.Alpha50) 

        #self.feat51, self.model51 = self.model_instance(self.Alpha51) 
        #self.feat34 = self.model_instance(self.Alpha34) 
        self.estimators = [('catboost1', self.model49), ('catboost2', self.model50)]
        #self.final_estimator = GradientBoostingRegressor(n_estimators=1000, subsample=0.85, min_samples_leaf=200, max_features=1, random_state=42)
        self.reg = VotingRegressor(self.estimators)#StackingRegressor(estimators=self.estimators,final_estimator=self.final_estimator)
        self.reg.fit(self.X_train, self.y_train)
        filename = 'finalized_model.sav'
        pickle.dump(self.reg, open(filename, 'wb'))


    def capture(self):
        """
        Capture alpha state.

        :return: dict or None if capturing is not supported
        """
        return None

    def restore(self, state):
        """
        Restore alpha state.

        :param state: the trained state to restore
        :return: True if state is restored, False otherwise
        """
        return False

    def predict_batch(self, items):
        """
        Predict function

        :param items: list of dictionaries
          * TIMEPERIODENDDATE: str 
          * UPC: str 
          * BASE_PRICE: float 
          * DISCOUNT_PERC: float 
          * AVGPCTACV: float 
          * AVGPCTACVANYDISPLAY: float 
          * AVGPCTACVANYFEATURE: float 
          * AVGPCTACVFEATUREANDDISPLAY: float 
          * AVGPCTACVTPR: float 
          * WEEK: int 
          * MONTH: int 
          * YEAR: int 
          * COUNTY: float 
          * STATE: float 
          * REGION: float 
          * C19_NYCASES: float 
          * C19_DEATHS: float 
          * INITIAL_CLAIMS: float 

        :return: list of predictions
        """
        print("Before predict_batch")
        #input_values = pd.DataFrame(items)

        X = pd.DataFrame(data=items)
        X['TIMEPERIODENDDATE'] = pd.to_datetime(X['TIMEPERIODENDDATE']).apply(
            lambda t: t.tz_localize(None))
            
        X = pd.merge(X, self.zillow, on=['TIMEPERIODENDDATE', 'COUNTY'],
                                   how="left")

        X = X.fillna(0)

        X = X.astype(
            {'BASE_PRICE': 'float64', 'DISCOUNT_PERC': 'float64', 'AVGPCTACV': 'float64',
             'AVGPCTACVANYDISPLAY': 'float64',
             'AVGPCTACVANYFEATURE': 'float64'
                , 'AVGPCTACVFEATUREANDDISPLAY': 'float64', 'AVGPCTACVTPR': 'float64', 'WEEK': 'int64'})

        # Convert string or object columns to categorical ones.
        for col, dtype in X.dtypes.to_dict().items():
            if str(dtype) in (['string', 'object']):
                X[col] = pd.Series(X[col], dtype='category')
                X[col] = pd.Series(X[col], dtype='category')

        print(X.dtypes)
        
        y = self.reg.predict(X)
        print(dict(zip(X.columns, X)))
        print("After predict_proba")
        return y


# Create alpha instance
model = Alpha()
fit = wqpt.fit(model.fit)
set_state = wqpt.set_state(model.set_state)
predict_batch = wqpt.predict_batch(model.predict_batch)
capture = wqpt.capture(model.capture)
restore = wqpt.restore(model.restore)

