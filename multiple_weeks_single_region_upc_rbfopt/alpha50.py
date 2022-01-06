import wqpt
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from collections import OrderedDict

# Summary function for a dataset


def big_df_get_summary(df):
    # initial summary
    ls_var = []
    ls_var_type = []
    ls_dist_val = []
    ls_mean = []
    ls_std = []
    ls_min = []
    ls_25 = []
    ls_50 = []
    ls_75 = []
    ls_max = []
    ls_pctmiss = []
    ls_sample = []

    ls_selected_col = [col for col in df.columns]

    # Describe for Numerical Columns
    df_num_desc = df.describe()

    # Value Counts for Categorical Columns

    # loop through each variables
    for col in df.columns:
        # get var and var type
        ls_var.append(col)
        ls_var_type.append(df[col].dtype)

        # calculate distinct value
        df_val_cnt = df[col].value_counts()
        i_dist_val = df_val_cnt.shape[0]
        ls_dist_val.append(i_dist_val)

        # calculate percentage missing
        percent_missing = df[col].isnull().sum() * 100 / len(df)
        ls_pctmiss.append(percent_missing)

        # get sample values
        ls_val = df_val_cnt.index.values
        i_num_sample = min([5, i_dist_val])
        str_sample = str(ls_val[0:i_num_sample])
        ls_sample.append(str_sample)

        if str(df[col].dtype) not in (['string', 'object', 'datetime64[ns]']):
            # get numerical stats
            ls_mean.append(df_num_desc.loc["mean", col])
            ls_std.append(df_num_desc.loc["std", col])
            ls_min.append(df_num_desc.loc["min", col])
            ls_25.append(df_num_desc.loc["25%", col])
            ls_50.append(df_num_desc.loc["50%", col])
            ls_75.append(df_num_desc.loc["75%", col])
            ls_max.append(df_num_desc.loc["max", col])
        else:
            # get categorical stats
            ls_mean.append(None)
            ls_std.append(None)
            ls_min.append(None)
            ls_25.append(None)
            ls_50.append(None)
            ls_75.append(None)
            ls_max.append(None)

    df_summary = pd.DataFrame(OrderedDict((
        ("variable", pd.Series(ls_var)),
        ("variable_type", pd.Series(ls_var_type)),
        ("n_distinct_values", pd.Series(ls_dist_val)),
        ("mean", pd.Series(ls_mean)),
        ("std", pd.Series(ls_std)),
        ("min", pd.Series(ls_min)),
        ("25p", pd.Series(ls_25)),
        ("50p", pd.Series(ls_50)),
        ("75p", pd.Series(ls_75)),
        ("max", pd.Series(ls_max)),
        ("pct_missing", pd.Series(ls_pctmiss)),
        ("sample_values", pd.Series(ls_sample))
    ))
    )
    return df_summary


drop_features = ["TIMEPERIODENDDATE", "DATE", "REGION_TYPE",
                 "REGION_NAME", "FULL_REGION_NAME", "DATE_TYPE", "WEEK", "UPC","MONTH", "COUNTY", "STATE", "REGION"]

cat_features = []#["UPC","MONTH", "COUNTY", "STATE", "REGION"]


def add_more_fts(data, add_fts):
    final = pd.merge(data, add_fts,  how='left', left_on=['TIMEPERIODENDDATE', 'COUNTY'], right_on=[
                      'DATE', "REGION_NAME"], suffixes=["", "_delete"])
    final = final.drop(
        columns=[col for col in list(final) if "_delete" in col])

    final = final.drop(columns=drop_features)
    for col in list(final):
        if col not in cat_features:
            final[col] = pd.Series(final[col], dtype='float64')

    return final


class Alpha:
    def __init__(self):
        zillow = pd.read_parquet('files/zillow.parquet')#.sample(100)
        zillow["DATE"] = pd.to_datetime(zillow["DATE"])
        zillow["MONTH"] = zillow["DATE"].dt.month
        zillow["YEAR"] = zillow["DATE"].dt.year
        zillow = zillow.drop_duplicates(
            subset=['REGION_NAME', 'REGION_TYPE', 'MONTH', 'YEAR'], keep='last')

        NYTcovid = pd.read_parquet('files/NYTcovid.parquet')
        unemployment = pd.read_parquet('files/unemployment.parquet')
        data = pd.read_parquet('files/file.parquet')
        NYTcovid_unemployment = pd.merge(NYTcovid, unemployment.drop(
            columns=["DATE_TYPE"]),  how='left', on=['WEEK', 'MONTH', 'YEAR', "STATE"])

        self.add_fts = pd.merge(NYTcovid_unemployment, zillow,  how='left', on=[
                                'REGION_NAME', 'REGION_TYPE', 'MONTH', 'YEAR'], suffixes=["", "_delete"])
        self.add_fts = self.add_fts.drop_duplicates(
            subset=['DATE', "REGION_NAME"], keep='last')

        self.add_fts = self.add_fts.drop(
            columns=[col for col in list(self.add_fts) if "_delete" in col])

        self.X_train = add_more_fts(data, self.add_fts)
        self.X_train = self.X_train.drop(columns=["SPPD"])
        print(list(self.X_train))
        self.y_train = data['SPPD']
     
                

    def set_state(self, ):
        pass

    def fit(self):
        self.list_imp_features = list(self.X_train)
  #      traindata = Pool(data=self.X_train, label=self.y_train,
   #                      cat_features=cat_features)
        self.model_base = CatBoostRegressor(cat_features = ['UPC', 'COUNTY', 'STATE', 'REGION'],
            iterations=10, loss_function="MAPE", l2_leaf_reg=0.05)
        
        '''
        self.model = CatBoostRegressor(
            iterations=10, loss_function="MAPE", l2_leaf_reg=0.05)
        self.model.fit(self.X_train, self.y_train)
        '''

    def capture(self):
        return None

    def restore(self, state):
        return False

    def predict_batch(self, items):

        '''
        y = self.model.predict(self.testdata) 
        '''# Catboost
        return 0#list(y)


# Create alpha instance
model = Alpha()
fit = wqpt.fit(model.fit)
set_state = wqpt.set_state(model.set_state)
predict_batch = wqpt.predict_batch(model.predict_batch)
capture = wqpt.capture(model.capture)
restore = wqpt.restore(model.restore)
