#Project to use various machine learning techniques to model credit card defaults in
#the case of default payments of credit card customers in Taiwan

#Libraries
import os
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

#Function to download credit card data
def download_credit_card_data():
    download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    data_path = "data_dir/"
    data_file_name = "credit_card_data.xlsx"
    #Making directory for data if one doesn't already exist
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    xlsx_path = os.path.join(data_path, data_file_name)
    if not os.path.isfile(xlsx_path):
        urllib.request.urlretrieve(download_url, xlsx_path)

def load_credit_card_data():
    data_path = "data_dir/"
    data_file_name = "credit_card_data.xlsx"
    xlsx_path = os.path.join(data_path, data_file_name)
    credit_card_file = pd.ExcelFile(xlsx_path)
    credit_card_df = credit_card_file.parse(skiprows=1)
    print(f"Reading in dataframe from {xlsx_path}")
    return credit_card_df

def show_histograms(df, features, bins = 50):
    df[features].hist(bins = bins, figsize=(20, 15))
    plt.show()

def split_train_test(df, test_train_ratio):
    shuffled_indices = np.random.permutation(len(df))
    test_set_length = int(len(df) * test_train_ratio)
    test_set_indices = shuffled_indices[:test_set_length]
    train_set_indices = shuffled_indices[test_set_length:]

    return df.iloc[train_set_indices], df.iloc[test_set_indices]

def add_age_category(df):
    df["AGE_cat"] = np.ceil(df["AGE"] / 10)
    df["AGE_cat"].where(df["AGE_cat"] < 7, 7.0, inplace = True)
    return df

def select_correlated_features(df, threshold = 0.1, plot_boolean = False):
    #Investigating correlation matrix
    correlation_mat = df.corr()
    correlation_mat_abs = abs(correlation_mat)
    correlation_mat_abs_mask = correlation_mat_abs["default payment next month"] > threshold
    # print(correlation_mat_abs["default payment next month"])
    # print(correlation_mat_abs_mask)
    # print(correlation_mat_abs["default payment next month"][correlation_mat_abs_mask])
    print(correlation_mat_abs["default payment next month"].sort_values(ascending = False))
    corr_features = correlation_mat_abs["default payment next month"][correlation_mat_abs_mask].index
    corr_features = np.array(corr_features)
    print(corr_features)
    if(plot_boolean):
        pd.plotting.scatter_matrix(df[corr_features])
        plt.show()

    #Only including correlated variables
    df = df[corr_features]
    return df

#Custom transform to handle pandas dataframe
class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Doc string for DataFrameSelector"""
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


#Making transformer class
class RatioAttributesAdder(BaseEstimator, TransformerMixin):
    """docstring for RatioAttributesAdder"""

    def __init__(self, add_payment_ratios = True):
        self.add_payment_ratios = add_payment_ratios
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        if self.add_payment_ratios:
            print("Adding payment ratios")
            print(X)
            print(X.shape)
            print(type(X))
            print(X[0,1])
            print(X[0,18])
            # LIMIT_BAL =
            payment_ratios = X[:,18] / X[:,1]
            return np.c_[X, payment_ratios]
        else:
            return np.c_[X]

#Implementing pipeline
prep_pipeline = Pipeline([
    # ('selector', DataFrameSelector(num_attributes))
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    # ('ratio_attribs_adder', RatioAttributesAdder()),#Adding on ratio variables
    ('std scaler', StandardScaler())
])


def main():
    print("hello world")

    #Downloading data
    download_credit_card_data()

    #Reading in data to pandas dataframe
    credit_card_df = load_credit_card_data()
    print(f"Columns: \n{credit_card_df.columns}\n")
    print(f"Summary stats:\n{credit_card_df.describe()}\n")
    print(f"Head:\n{credit_card_df.head()}\n")

    #Look at histograms of data

    #Choose interesting features to look at
    features = ['LIMIT_BAL','SEX', 'EDUCATION', 'AGE']

    # show_histograms(credit_card_df, features)
    # show_histograms(credit_card_df, credit_card_df.columns)

    #Seems to be some strangely popular ages in the data
    #University seems to be the most popular level of education
    #BILL_AMT features are very tail-heavy

    #Create test and train sets with random number generator seed set
    np.random.seed(42)
    train_set, test_set = split_train_test(credit_card_df, 0.2)
    print(f"Train set length = {len(train_set)}")
    print(f"Test set length = {len(test_set)}")

    #Shall sample from age strata to ensure groups are representative of age groups
    #Dividing by 10 gives 6 age categories, rounding up to five groups
    credit_card_df = add_age_category(credit_card_df)
    # show_histograms(credit_card_df, ["AGE_cat", "AGE"])

    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    for train_index, test_index in split.split(credit_card_df, credit_card_df["AGE_cat"]):
        strat_train_set = credit_card_df.loc[train_index]
        strat_test_set = credit_card_df.loc[test_index]

    print(f"Strat train set length = {len(strat_train_set)}")
    print(f"Strat test set length = {len(strat_test_set)}")

    #Checking how age proportionalities match up with random or stratified sampling

    train_set = add_age_category(train_set)
    strat_train_set = add_age_category(strat_train_set)

    print("Overall:")
    print(credit_card_df["AGE_cat"].value_counts() / len(credit_card_df))
    print("Random:")
    print(train_set["AGE_cat"].value_counts() / len(train_set))
    print("Stratified:")
    print(strat_train_set["AGE_cat"].value_counts() / len(strat_train_set))
    #Stratified sampling does give a better representation of the overall data

    #Removing AGE_cat variable from data frames
    for set_ in (strat_test_set, strat_train_set):
        set_.drop("AGE_cat", axis = 1, inplace = True)

    # print("Overall\tRandom\tStrat\tRand. Error\tStrat. Error\n")
    # for i, cat in enumerate(credit_card_df["AGE_cat"].value_counts()/len(credit_card_df)):
    #     print(cat,
    #     train_set["AGE_cat"].value_counts()[i]/len(train_set),
    #     strat_train_set["AGE_cat"].value_counts()[i]/len(strat_train_set))
    #     print((train_set["AGE_cat"].value_counts()[i]/len(train_set) - cat) * 100/cat, (strat_train_set["AGE_cat"].value_counts()[i]/len(strat_train_set) - cat) * 100/cat)

    #Making a sample of the training set to experiment with
    strat_train_set_sample = strat_train_set.sample(frac = 0.5, random_state = 42)


    #Trying to get attribute adder to work
    attr_adder = RatioAttributesAdder()
    extra_attribs = attr_adder.transform(strat_train_set_sample.values)


    return

    strat_train_set_sample_array = prep_pipeline.fit_transform(strat_train_set_sample)
    print(strat_train_set_sample_array)
    print(strat_train_set_sample_array.shape)
    #Putting it back in to pandas df
    strat_train_set_sample = pd.DataFrame(strat_train_set_sample_array, columns = strat_train_set_sample.columns)
    print(strat_train_set_sample)




    #Do I need to add extra variables?
    #Could try adding total credit dvidied by first payment and see if there's any correlation
    strat_train_set_sample["PAY_AMT1_ratio"] = strat_train_set_sample["PAY_AMT1"] / strat_train_set_sample["LIMIT_BAL"]
    #Doing this for every PAY_AMT
    for col in strat_train_set_sample.loc[:, "PAY_AMT1":"PAY_AMT6"]:
        new_column_name = str(col) + ("_ratio")
        print(col, new_column_name)
        strat_train_set_sample[new_column_name] = strat_train_set_sample[col] / strat_train_set_sample["LIMIT_BAL"]
    # print(strat_train_set_sample.loc[:, "PAY_AMT1":"PAY_AMT6"])
    print(strat_train_set_sample.columns)



    strat_train_set_sample = select_correlated_features(strat_train_set_sample, threshold = 0.02, plot_boolean = False)

    #Check for missing data values
    print(strat_train_set_sample.info())
    print(strat_train_set_sample.describe())
    #12000 values in each - no missing values, but shall add
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_median.fit(strat_train_set_sample)
    imp_median.transform(strat_train_set_sample)
    #Any values need to be encoded?


    #Feature scaling using StandardScaler
    scaler = StandardScaler().fit(strat_train_set)
    print(scaler.mean_)
    print(scaler.scale_)





main()
