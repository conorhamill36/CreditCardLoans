#Project to use various machine learning techniques to model credit card defaults in
#the case of default payments of credit card customers in Taiwan

#Libraries
import os
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    train_set, test_set = split_train_test(credit_card_df, 0.2)
    print(f"Train set length = {len(train_set)}")
    print(f"Test set length = {len(test_set)}")

    #Show correlation matrix


    #Do I need to add extra variables?


    #Check for missing data values



    #Any values need to be encoded?


main()
