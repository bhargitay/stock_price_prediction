# -- coding: utf-8 --
import pandas as pd
import numpy as np
import os
import zipfile
import matplotlib.pyplot as plt
import math


def read_huge_market_stock_data(zip_file_path: str, company_ticker: str, date_interval: list):
    """
    Reads in the data from the 'Huge Market Stock Dataset' based on the filters.

    :param zip_file_path: path to huge_stock_market_data.zip
    :param company_ticker: company ticker
    :param date_interval: date intervals you want to filter the dataset by.
                          You should use the following format: [first_date, last_date].

    :return: A pandas Dataframe with the filtered data from the 'Huge Stock Market dataset'.
    """
    # unzipping huge_stock_market_data.zip
    # if data is already unzipped then do not unzip it again
    result_path = zip_file_path[:-4]
    if not os.path.exists(result_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(result_path)
    else:
        print('File is already unzipped')

    # save the parent directory to a variable, because we are going to need it later
    parent_dir = os.getcwd()
    # change the directory to the path of the ETF files
    os.chdir(result_path + '/Data/Stocks/')

    # read in file belonging to the company ticker
    if os.path.getsize(company_ticker) > 0:
        df_stock_market = pd.read_csv(company_ticker)
    else:
        print('File is empty')

    # change the working directory back to parent directory
    os.chdir(parent_dir)

    df_stock_market.reset_index(inplace=True, drop=True)

    # convert 'Date' to datetime
    df_stock_market['Date'] = pd.to_datetime(df_stock_market['Date'], format='%Y-%m-%d')

    # filtering date range according to the date_interval parameter
    df_stock_market = df_stock_market[(df_stock_market.Date >= date_interval[0]) &
                                      (df_stock_market.Date <= date_interval[1])]

    #df_stock_market.sort_index(inplace=True)
    df_stock_market.set_index("Date", inplace=True)

    return df_stock_market


def plot_closing_price_history(close_price_col: pd.Series, company_name: str):
    """
    Plots closing price history.

    :param close_price_col: pd.Series containing the closing price
    :param company_name: name of company
    :return: A plot of closing price history.
    """
    plt.figure(figsize=(16, 8))
    plt.title(company_name + ' - Closing Price History', fontsize=18)
    plt.plot(close_price_col)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price (USD)', fontsize=12)


def plot_closing_price_and_EMA(closing_price: pd.Series, EMA: pd.Series):
    """
    Plots closing price and Exponential Moving Average (EMA).

    :param closing_price: pd.Series containing the closing prices
    :param EMA: pd.Series containing the EMA
    :return: A plot of Closing pPrice and Exponential Moving Average over a period of time.
    """
    plt.figure(figsize=(16, 8))
    plt.plot(closing_price, label='Closing Price')
    plt.plot(EMA, label='EMA')
    plt.title('Closing Price vs Exponential Moving Average', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price (USD)', fontsize=12)
    plt.legend()


def create_train_test_split(training_data_size: float, training_days: int, data: np.array, scaled_data):
    """
    Splits the dataset into train and test sets.

    :param training_data_size: represent the proportion of the dataset to include in the train split.
                               Should be between 0.0 and 1.0.
    :param training_days: number of days we we want to predict the next day's stock price upon
    :param data: a np.array containing all the closing prices
    :param scaled_data: scaled data
    :return: the Length of training data and the split train and test set
    """
    # Compute the number of rows to train the model on
    training_data_len = math.ceil(len(scaled_data) * training_data_size)
    # train set
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(training_days, len(train_data)):
        x_train.append(train_data[i - training_days:i, 0])
        y_train.append(train_data[i, 0])

    # test set
    test_data = scaled_data[training_data_len - training_days:, :]
    y_test = data[training_data_len:, :]
    x_test = []
    for i in range(training_days, len(test_data)):
        x_test.append(test_data[i - training_days:i, 0])

    return training_data_len, x_train, y_train, x_test, y_test


def create_3d_arrays(x: list):
    """
    Creates 3D arrays from input values.

    :param x: list of input arrays
    :return: 3D arrays based on list of arrays for inputs
    """
    # convert list of arrays to numpy arrays
    x = np.array(x)
    # create the 3D array for LSTM
    x = np.reshape(x, (x.shape[0], x.shape[1],1))
    return x


def plot_result(dataset: pd.DataFrame, training_data_len: int, predictions: np.array):
    """
    Plots the predictions of our model.

    :param dataset: pd.DataFrame containing only the closing price data
    :param training_data_len: length of training data - number of rows used for training the model for a given period
    :param predictions: predictions made by our model.
    :return: A plot with the real and predicted values.
    """
    # Plot/Create the data for the graph
    train_close = dataset[:training_data_len][['Close']]
    valid = dataset[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('LSTM Predictions', fontsize=18)
    plt.plot(train_close, 'tab:blue')
    plt.plot(valid['Close'], linestyle=':', color='tab:blue')
    plt.plot(valid['Predictions'], 'tab:orange')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price (USD)', fontsize=12)
    plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
