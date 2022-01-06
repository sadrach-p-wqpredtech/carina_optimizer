import wqpt
import pandas as pd
from catboost import CatBoostRegressor
from pandas.tseries.offsets import *


class Alpha:
    def __init__(self):
        pd.set_option('display.max_columns', None)
        self.train_data = pd.read_parquet('files/file.parquet')#.sample(100)
        self.train_data['TIMEPERIODENDDATE'] = pd.to_datetime(self.train_data['TIMEPERIODENDDATE']).apply(lambda t: t.tz_localize(None))

        print("TRAIN_DATA ORIGIN: ")
        print(self.train_data)

        self.zillow = pd.read_parquet('files/zillow.parquet')
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
        # Convert string or object columns to categorical ones.
        self.cat_features = []
        for col, dtype in self.X_train.dtypes.to_dict().items():
            if str(dtype) in (['string', 'object']):
                self.X_train[col] = pd.Series(self.X_train[col], dtype='category')
                self.X_train[col] = pd.Series(self.X_train[col], dtype='category')
                #self.cat_features.append(col)

        self.y_train = self.train_data['SPPD']
        #print(self.X.dtypes)
        pass

    def set_state(self, ):
        # Example:
        # self.observed_states.append([])
        pass

    def fit(self):
        self.model_base = CatBoostRegressor(cat_features = ['UPC', 'COUNTY', 'STATE', 'REGION'],
                                       loss_function='MAPE')
        
        '''
        self.model = CatBoostRegressor(cat_features= self.cat_features,
                                       loss_function='MAPE')
        self.model.fit(self.X_train, self.y_train)
        '''
        pass

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
        '''
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
        
        y = self.model.predict(X)
        print(y)
        '''
        return 0#list(y)


# Create alpha instance
model = Alpha()
fit = wqpt.fit(model.fit)
set_state = wqpt.set_state(model.set_state)
predict_batch = wqpt.predict_batch(model.predict_batch)
capture = wqpt.capture(model.capture)
restore = wqpt.restore(model.restore)
