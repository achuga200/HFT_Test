# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:21:33 2020

@author: Fidelis Achu
"""

# Import Dependencies
from cointegration_analysis import estimate_long_run_short_run_relationships, engle_granger_two_step_cointegration_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def read_data(filename):
    ''''This function reads the .csv stored at the 'filename' location and returns a DataFrame
    with two levels of column names. The first level column contains the Stock Name and the 
    second contains the type of market data, e.g. bid/ask, price/volume.
    '''
    df  = pd.read_csv(filename,index_col = 0)
    df.columns = [df.columns.str[-2:], df.columns.str[:-3]] # to move the stock up and down the column
    
    return df

market_data = read_data('pairs Trading.csv')

# Show the First 5 Rows
print(market_data.head())

# Show the Stocks
stock_names = list(market_data.columns.get_level_values(0).unique())
print('The stocks available are',stock_names)
    


market_data_segmented  = market_data[:250]

def bid_ask_price_plot(stock1,stock2):
    '''
    This function creates a subplot with a specified gridsize to be able to
    effectively match it with a different subplot while still maintaining
    it's independency of being able to just show this plot.
    '''
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    plt.title('Bid & Ask Prices Development of the Stocks ' + stock1 + " and " + stock2)
    plt.grid()
    
    ax1.plot(market_data_segmented.index,
             market_data_segmented[stock1, 'BidPrice'])
    ax1.plot(market_data_segmented.index,
             market_data_segmented[stock1, 'AskPrice'])
    
    ax1.plot(market_data_segmented.index,
             market_data_segmented[stock2, 'BidPrice'])
    ax1.plot(market_data_segmented.index,
             market_data_segmented[stock2, 'AskPrice'])
    
     # We don't want to see all the timestamps
    ax1.axes.get_xaxis().set_visible(False)

    ax1.legend([stock1 + " Bid Price", stock1 + " Ask Price", stock2 + " Bid Price", stock2 + " Ask Price"], loc='upper right')
    
    
    
def bid_ask_volume_plot(stock1,stock2):
    '''
    This function is very similar to above's function with the exception
    of creating a smaller subplot and using different data. This function
    is meant for displaying volumes. '''
    
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1)
    plt.title('Bid & Ask Volumes Development of the Stocks ' + stock1 + " and " + stock2)
    plt.grid()

    ax2.plot(market_data_segmented.index,
             market_data_segmented[stock1, 'BidVolume'])
    ax2.plot(market_data_segmented.index,
             market_data_segmented[stock1, 'AskVolume'])

    ax2.plot(market_data_segmented.index,
             market_data_segmented[stock2, 'BidVolume'])
    ax2.plot(market_data_segmented.index,
             market_data_segmented[stock2, 'AskVolume'])

    # We don't want to see all the timestamps
    ax2.axes.get_xaxis().set_visible(False)

    ax2.legend([stock1 + " Bid Volume", stock1 + " Ask Volume", stock2 + " Bid Volume", stock2 + " Ask Volume"], loc='upper right')

# Show Plot
plt.figure(figsize=(15, 15))
plt.show(bid_ask_price_plot("CC", "MM"), bid_ask_volume_plot("CC", "MM"))