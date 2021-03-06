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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

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

#Function to turn credit card data in to a DataFrame
def load_credit_card_data():
    data_path = "data_dir/"
    data_file_name = "credit_card_data.xlsx"
    xlsx_path = os.path.join(data_path, data_file_name)
    credit_card_file = pd.ExcelFile(xlsx_path)
    credit_card_df = credit_card_file.parse(skiprows=1)
    print(f"Reading in dataframe from {xlsx_path}")
    return credit_card_df

#Displays the histograms of selected features in a dataset
def show_histograms(df, features, bins = 50):
    df[features].hist(bins = bins, figsize=(20, 15))
    plt.show()

#Splits a data set in to train and test sets, with ratio defined
def split_train_test(df, test_train_ratio = 0.8):
    shuffled_indices = np.random.permutation(len(df))
    test_set_length = int(len(df) * test_train_ratio)
    test_set_indices = shuffled_indices[:test_set_length]
    train_set_indices = shuffled_indices[test_set_length:]

    return df.iloc[train_set_indices], df.iloc[test_set_indices]

def add_age_category(df):
    df["AGE_cat"] = np.ceil(df["AGE"] / 10)
    df["AGE_cat"].where(df["AGE_cat"] < 7, 7.0, inplace = True)
    return df

def select_correlated_features(df, threshold = 0.1, plot_boolean = False, target = "default payment next month"):
    #Investigating correlation matrix
    correlation_mat = df.corr()
    correlation_mat_abs = abs(correlation_mat)
    correlation_mat_abs_mask = correlation_mat_abs["default payment next month"] > threshold
    print(correlation_mat)
    print(correlation_mat_abs["default payment next month"])
    # print(correlation_mat_abs_mask)
    print(correlation_mat_abs["default payment next month"][correlation_mat_abs_mask])
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


#Making transformer class that adds on another attribute to data frame
class RatioAttributesAdder(BaseEstimator, TransformerMixin):
    """Class to add the ratio of the first payment divided
    by the total payment to the dataframe"""

    def __init__(self, add_payment_ratios = True):
        self.add_payment_ratios = add_payment_ratios
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        if self.add_payment_ratios:
            print("Adding payment ratios")
            print("X: ")
            print(X)
            print("X shape: ")
            print(X.shape)
            print("X type: ")
            print(type(X))
            print("X[0,1]: ")
            print(X[0,1])
            print("X[0,18]: ")
            print(X[0,18])
            print("X[:,1]")
            print(X[:,1])
            print("len X[:,18]")
            print(len(X[:,18]))
            # LIMIT_BAL =
            payment_ratios = X[:,18] / X[:,1]
            print(f"shape(payment_ratios: {payment_ratios.shape}")
            print(f"shape(payment_ratios: {np.c_[X].shape}")
            print(f"shape(payment_ratios: {np.c_[X, payment_ratios].shape}")
            #Returns 2D numpy array
            return np.c_[X, payment_ratios]
        else:
            return np.c_[X]

#Implementing pipeline for preparation of data
prep_pipeline = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('ratio_attribs_adder', RatioAttributesAdder()),#Adding on ratio variables
    ('std scaler', StandardScaler())
    # ('selector', DataFrameSelector(num_attributes))
])


def main():
    print("hello world")

    #Downloading data
    print("Downloading data")
    download_credit_card_data()

    #Reading in data to pandas dataframe
    print("Reading in data to pandas dataframe")
    credit_card_df = load_credit_card_data()
    print(f"Columns: \n{credit_card_df.columns}\n")
    print(f"Summary stats:\n{credit_card_df.describe()}\n")
    print(f"Head:\n{credit_card_df.head()}\n")

    #Look at histograms of data

    #Choosing interesting features to look at
    print("Choosing interesting features to look at")
    features = ['LIMIT_BAL','SEX', 'EDUCATION', 'AGE']

    #Create test and train sets with random number generator seed set
    print("Create test and train sets with random number generator seed set")
    np.random.seed(42)
    train_set, test_set = split_train_test(credit_card_df, 0.2)
    print(f"Train set length = {len(train_set)}")
    print(f"Test set length = {len(test_set)}")
    #Test data set is being set to the side for the time being
    #...but I also need to transform it at some point as well

    #Seems to be some strangely popular ages in the data
    #University seems to be the most popular level of education
    #BILL_AMT features are very tail-heavy


    #Check for missing data values
    # print("Check for missing data values")
    # print(strat_train_set_sample.info())
    # print(strat_train_set_sample.describe())
    # #12000 values in each - no missing values, but shall add
    # print("Adding imputer")
    # imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    # imp_median.fit(strat_train_set_sample)
    # imp_median.transform(strat_train_set_sample)
    # #Any values need to be encoded?
    #
    #
    # #Feature scaling using StandardScaler
    # print("Feature scaling using StandardScaler")
    # scaler = StandardScaler().fit(strat_train_set)
    # print(scaler.mean_)
    # print(scaler.scale_)




    #Shall sample from age strata to ensure groups are representative of age groups
    print("Shall sample from age strata to ensure groups are representative of age groups")
    #Dividing by 10 gives 6 age categories, rounding up to five groups
    print("Dividing by 10 gives 6 age categories, rounding up to five groups")
    credit_card_df = add_age_category(credit_card_df)


    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    for train_index, test_index in split.split(credit_card_df, credit_card_df["AGE_cat"]):
        strat_train_set = credit_card_df.loc[train_index]
        strat_test_set = credit_card_df.loc[test_index]

    print(f"Strat train set length = {len(strat_train_set)}")
    print(f"Strat test set length = {len(strat_test_set)}")

    #Checking how age proportionalities match up with random or stratified sampling
    print("Checking how age proportionalities match up with random or stratified sampling")
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
    print("Removing AGE_cat variable from data frames")
    for set_ in (strat_test_set, strat_train_set):
        set_.drop("AGE_cat", axis = 1, inplace = True)

    # print("Overall\tRandom\tStrat\tRand. Error\tStrat. Error\n")
    # for i, cat in enumerate(credit_card_df["AGE_cat"].value_counts()/len(credit_card_df)):
    #     print(cat,
    #     train_set["AGE_cat"].value_counts()[i]/len(train_set),
    #     strat_train_set["AGE_cat"].value_counts()[i]/len(strat_train_set))
    #     print((train_set["AGE_cat"].value_counts()[i]/len(train_set) - cat) * 100/cat, (strat_train_set["AGE_cat"].value_counts()[i]/len(strat_train_set) - cat) * 100/cat)


    #Splitting test set in to target and feature variables
    strat_test_set_X = strat_test_set.drop(columns = ["default payment next month"])
    strat_test_set_y = strat_test_set["default payment next month"]

    #Adding ratio variable to test set
    #Instance of attribute adder
    attr_adder = RatioAttributesAdder(add_payment_ratios = True)
    #Returns 2D numpy array
    extra_attribs = attr_adder.transform(strat_test_set_X.values)
    #Adding new data to dataframe
    strat_test_set_X = strat_test_set_X.assign(Ratio = extra_attribs[:,-1])


    #Making a sample of the training set to experiment with
    print("Making a sample of the training set to experiment with")
    strat_train_set_sample = strat_train_set.sample(frac = 0.9, random_state = 42)


    #Adding another attribute to data frame
    # print("Adding another attribute to data frame")
    # #Instance of attribute adder
    # attr_adder = RatioAttributesAdder(add_payment_ratios = True)
    # #Returns 2D numpy array
    # extra_attribs = attr_adder.transform(strat_train_set_sample.values)
    # #Adding new data to dataframe
    # strat_train_set_sample = strat_train_set_sample.assign(Ratio = extra_attribs[:,-1])
    #
    # print("Shapes:")
    # print(extra_attribs.shape)
    # print(strat_train_set_sample.shape)
    #
    # extra_attribs_columns = (strat_train_set_sample.columns)
    # print(extra_attribs_columns)
    # print(type(extra_attribs_columns))
    #
    # # strat_train_set_sample = pd.DataFrame(data = extra_attribs, columns = strat_train_set_sample.columns)
    #
    #
    # print(type(extra_attribs))
    # print(type(strat_train_set_sample))
    # print(len(extra_attribs))
    # print(len(strat_train_set_sample))

    print(strat_train_set_sample.head())


    #Using preparation pipeline
    print("Using preparation pipeline")
    print(strat_train_set_sample.columns)
    strat_train_set_sample_X = strat_train_set_sample.drop(columns = ["default payment next month"])
    strat_train_set_sample_y = strat_train_set_sample["default payment next month"]

    print(strat_train_set_sample_X.head())
    print(strat_train_set_sample_y.head())

    strat_train_set_sample_X_array = prep_pipeline.fit_transform(strat_train_set_sample_X)
    print(strat_train_set_sample_X_array)
    print(strat_train_set_sample_X_array.shape)
    #Putting it back in to pandas df
    new_columns = list(strat_train_set_sample_X.columns)
    new_columns.append("Ratio")
    print(new_columns)
    strat_train_set_sample_X = pd.DataFrame(columns = new_columns, data = strat_train_set_sample_X_array)
    # strat_train_set_sample.assign(Ratio = [])
    # strat_train_set_sample.append(strat_train_set_sample_array)

    print(strat_train_set_sample_X)
    print(strat_train_set_sample_X.describe())



    #Selecting correlated features
    print("Selecting correlated features using earlier data ")
    strat_train_set_sample = select_correlated_features(strat_train_set_sample, threshold = 0.08, plot_boolean = False, target = "default payment next month")

    print(strat_train_set_sample)





    #Now trying some models for the data
    print("Now trying some models for the data")

    print(strat_train_set_sample["PAY_0"])


    #Logistic regression
    print("\n\n\nLogistic regression")
    #Instance of logistic regression model
    log_reg = LogisticRegression()#penalty = 'l2', C = 0.1,random_state = 0)

    # log_reg.fit(X, y)
    # print(f"Score: {log_reg.score(X, y)}")

    #Using GridSearchCV to find optimum parameters

    #Making parameter grid
    param_grid = [
    {
    'C' : np.logspace(-5, 1, 4),
    'penalty' : ['l2'],
    'solver' : ['sag']
    }
    ]

    #Making grid search object
    grid_clf = GridSearchCV(log_reg, param_grid = param_grid, cv=3, scoring='neg_mean_squared_error')

    grid_clf.fit(strat_train_set_sample_X, strat_train_set_sample_y)

    print(grid_clf.best_params_)

    cvres = grid_clf.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(mean_score, params)

    log_reg = LogisticRegression(penalty = 'l2', C = 0.001, random_state = 0)
    log_reg.fit(strat_train_set_sample_X, strat_train_set_sample_y)
    print(f"Score: {log_reg.score(strat_test_set_X, strat_test_set_y)}")

    #Decision trees
    print("Decision tree")

    #Instance of decision tree classifier
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(strat_train_set_sample_X, strat_train_set_sample_y)

    print(f"Score: {dt_clf.score(strat_test_set_X, strat_test_set_y)}")

    #Support Vector Machine - is taking a very very long time
    # X_svm = np.array(strat_train_set_sample["PAY_0"]).reshape(-1, 1)#.reshape(-1, 1)
    # print("Support Vector Machine")
    # svm_clf = svm.SVC(kernel='linear')
    # svm_clf.fit(X_svm[:1000], y[:1000])
    # print(f"Score: {svm_clf.score(X_svm, y)}")


    #K-nearest neighbours model
    print("K-nearest neighbours")

    knn_model = KNeighborsClassifier(n_neighbors = 4)
    knn_model.fit(strat_train_set_sample_X, strat_train_set_sample_y)
    print(f"Score: {knn_model.score(strat_test_set_X, strat_test_set_y)}")


    #Random Forest Regressor
    print("Random forest regressor")
    param_grid = [
        {'bootstrap': [True], 'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]

    forest_reg = RandomForestClassifier()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
        scoring='neg_mean_squared_error')

    grid_search.fit(strat_train_set_sample_X, strat_train_set_sample_y)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    print("Checking that the lower the gridsearchcv score, the better")

    forest_reg_1 = RandomForestClassifier(max_features = 8, n_estimators = 30)
    forest_reg_1.fit(strat_train_set_sample_X, strat_train_set_sample_y)
    print(f"Score: {forest_reg_1.score(strat_test_set_X, strat_test_set_y)}")

    forest_reg_2 = RandomForestClassifier(max_features = 8, n_estimators = 3)
    forest_reg_2.fit(strat_train_set_sample_X, strat_train_set_sample_y)
    print(f"Score: {forest_reg_2.score(strat_test_set_X, strat_test_set_y)}")

    forest_reg_3 = RandomForestClassifier(max_features = 2, n_estimators = 3)
    forest_reg_3.fit(strat_train_set_sample_X, strat_train_set_sample_y)
    print(f"Score: {forest_reg_3.score(strat_test_set_X, strat_test_set_y)}")



    #Making an ensemble model
    print(f"Making an ensemble model")
    voting_clf = VotingClassifier(
        estimators = [('lr', log_reg), ('knn', knn_model), ('rndm_for', forest_reg_3)],# ('svm', svm_clf)],
        voting = 'soft' #Soft currently doing better than hard
    )
    voting_clf.fit(strat_train_set_sample_X, strat_train_set_sample_y)

    print(f"Score: {voting_clf.score(strat_test_set_X, strat_test_set_y)}")


    return




main()
