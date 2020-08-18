import pandas as pd
import numpy as np
import os
import zipfile
import matplotlib.pyplot as plt
import math
import datetime


def read_huge_market_stock_data(zip_file_path, company_ticker, date_interval):
    # unzipping huge_stock_market_data.zip
    # if data is already unzipped then do not unzip it again
    result_path = zip_file_path[:-4]
    if not os.path.exists(result_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(result_path)
    else:
        print('File is already unzipped')

    # we save the parent directory to a variable, because we are going to need it later
    parent_dir = os.getcwd()
    # we change the directory to the path ot the ETF files
    os.chdir(result_path + '/Data/Stocks/')

    # read in file belonging to the company
    if os.path.getsize(company_ticker) > 0:
        df_stock_market = pd.read_csv(company_ticker)
    else:
        print('File is empty')

    # change the working directory back to parent directory
    os.chdir(parent_dir)

    df_stock_market.reset_index(inplace=True, drop=True)

    # convert 'Date' to datetime object
    df_stock_market['Date'] = pd.to_datetime(df_stock_market['Date'], format='%Y-%m-%d') # df_stock_market['Date'].dt.date()

    # check number of rows before date filtering
    # print("Number of rows in the dataset before filtering: ", df_stock_market.shape[0])

    # filtering date range according to the instructions
    df_stock_market = df_stock_market[(df_stock_market.Date >= date_interval[0]) &
                                      (df_stock_market.Date <= date_interval[1])]
    # check number of rows after date filtering
    # print("Number of rows in the dataset after filtering: ", df_stock_market.shape[0])

    # set index to 'Date'
    #df_stock_market = df_stock_market.set_index('Date')  # .sort_index(inplace=True)
    df_stock_market.sort_index(inplace=True)

    return df_stock_market


def plot_closing_price_history(close_price_col, company_name):
    # Visualize the closing price history
    plt.figure(figsize=(16, 8))
    plt.title(company_name + ' - Closing Price History')
    plt.plot(close_price_col)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    # plt.show()


def create_train_test_split(training_data_size, training_days, dataset, scaled_data):
    # Get /Compute the number of rows to train the model on
    training_data_len = math.ceil(len(scaled_data) * training_data_size)
    # Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(training_days, len(train_data)):
        x_train.append(train_data[i - training_days:i, 0])
        y_train.append(train_data[i, 0])

    # Test data set
    test_data = scaled_data[training_data_len - training_days:, :]
    # Create the x_test and y_test data sets
    x_test = []
    y_test = dataset[training_data_len:,
             :]  # Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
    for i in range(training_days, len(test_data)):
        x_test.append(test_data[i - training_days:i, 0])

    return training_data_len, x_train, y_train, x_test, y_test

def create_3D_arrays(X, y):
    #Convert x_train and y_train to numpy arrays
    X, y = np.array(X), np.array(y)
    #Reshape the data into the shape accepted by the LSTM
    X = np.reshape(X, (X.shape[0],X.shape[1],1))
    return X, y

def plot_result(data, training_data_len, predictions):
    # Plot/Create the data for the graph
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
