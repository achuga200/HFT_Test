# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:06:17 2020

@author: Fidelis Achu
"""
from __future__ import print_function
from cointegration_analysis import estimate_long_run_short_run_relationships, engle_granger_two_step_cointegration_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt







################# DATA ANALYSIS SECTION #############################

def read_data(filename):
    ''''This function reads the .csv stored at the 'filename' location and returns a DataFrame
    with two levels of column names. The first level column contains the Stock Name and the 
    second contains the type of market data, e.g. bid/ask, price/volume.
    '''
    df  = pd.read_csv(filename,index_col = 0)
    df.columns = [df.columns.str[-2:], df.columns.str[:-3]] # to move the stock up and down the column
    
    return df

market_data = read_data('Data/pairs Trading.csv')

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
#plt.figure(figsize=(15, 15))
#plt.show(bid_ask_price_plot("CC", "MM"), bid_ask_volume_plot("CC", "MM"))
    
    
    
for stock in stock_names:
    market_data[stock,'MidPrice'] = (market_data[stock,'BidPrice'] + market_data[stock,'AskPrice']) / 2
    market_data = market_data.sort_index(axis=1) # 1 = column and 0 = rows to sort
    
#market_data
    
    

def mid_price_check(stock):
    '''
    Function that checks for different stocks if the MidPrice
    is correctly specified.
    '''
    plt.figure(figsize=(20, 5))
    plt.plot(market_data[stock,'AskPrice'][:100])
    plt.plot(market_data[stock,'MidPrice'][:100])
    plt.plot(market_data[stock,'BidPrice'][:100])

    plt.xticks([]) # Timestamp is not Important
    plt.title('Ask, Bid and Mid Price Development of Stock ' + stock)
    plt.legend(["Ask Price", "Mid Price", "Bid Price"], loc='lower left')
    plt.show()
    
#mid_price_check('MM')



# Obtain the statistical parameters for each and every pair
data_analysis = {'Pairs': [],
                 'Constant': [],
                 'Gamma': [],
                 'Alpha': [],
                 'P-Value': []}

data_zvalues = {}

for stock1 in stock_names:
    for stock2 in stock_names:
        if stock1 != stock2:
            if (stock2, stock1) in data_analysis['Pairs']:
                continue

            pairs = stock1, stock2
            constant = estimate_long_run_short_run_relationships(np.log(
                market_data[stock1, 'MidPrice']), np.log(market_data[stock2, 'MidPrice']))[0]
            gamma = estimate_long_run_short_run_relationships(np.log(
                market_data[stock1, 'MidPrice']), np.log(market_data[stock2, 'MidPrice']))[1]
            alpha = estimate_long_run_short_run_relationships(np.log(
                market_data[stock1, 'MidPrice']), np.log(market_data[stock2, 'MidPrice']))[2]
            pvalue = engle_granger_two_step_cointegration_test(np.log(
                market_data[stock1, 'MidPrice']), np.log(market_data[stock2, 'MidPrice']))[1]
            zvalue = estimate_long_run_short_run_relationships(np.log(
                market_data[stock1, 'MidPrice']), np.log(market_data[stock2, 'MidPrice']))[3]

            data_analysis['Pairs'].append(pairs)
            data_analysis['Constant'].append(constant)
            data_analysis['Gamma'].append(gamma)
            data_analysis['Alpha'].append(alpha)
            data_analysis['P-Value'].append(pvalue)

            data_zvalues[pairs] = zvalue
            
data_analysis = round(pd.DataFrame(data_analysis),4).set_index('Pairs')






# Visualize the P-values
def plot_pvalues():
    """
    This function plots all obtained P-values.
    """
    plt.figure(figsize=(20, 5))
    plt.hist(data_analysis['P-Value'], bins=100)
    plt.xlabel('P-value')
    plt.ylabel('Number of observations')
    plt.title('All obtained P-values')
    plt.show()

#plot_pvalues()
    
# Show Top 10 and Bottom 10
#display(data_analysis.sort_values('P-Value')[:10])
#display(data_analysis.sort_values('P-Value')[-10:])
    
    
# Selecting tradable pairs where P-Value < 0.01 and create a seperate DataFrame containing these pairs
tradable_pairs_analysis = data_analysis[data_analysis['P-Value'] < 0.01].sort_values('P-Value')

#tradable_pairs_analysis



# Get all the tradable stock pairs into a list
stock_pairs = list(tradable_pairs_analysis.index.values.tolist())

# Show the Pairs
#stock_pairs








############# ALGORITHM SECTION ###########################

# Create a list of unique tradable stocks
list_stock1 = [stock[0] for stock in stock_pairs]
list_stock2 = [stock[1] for stock in stock_pairs]

for stock in list_stock2:
    list_stock1.append(stock)
    
unique_stock_list = list(set(list_stock1))

# Create a new DataFrame containing all market information for the tradable pairs
tradable_pairs_data = market_data[unique_stock_list]
#tradable_pairs_data.head()



def Plot_Tradable_Z():
    """
    This function plots the z-values of all pairs based on
    the data_zvalues dataframe.
    """
    for pair in stock_pairs: 
        zvalue = data_zvalues[pair]
        plt.figure(figsize=(20,5))
        plt.title('Error-correction term stock pair {}'.format(pair))
        zvalue.plot()
        plt.xlabel('Time')
        plt.ylabel('Magnitude')

        xmin = 0
        xmax = len(zvalue)
        plt.hlines(0.005, xmin, xmax, 'g') # Note 0.005 is randomly chosen
        plt.hlines(-0.005, xmin, xmax, 'r') # Note -0.005 is randomly chosen
        
        plt.legend(['Z-Value', 'Positive Threshold', 'Negative Threshold'], loc='lower left')
        
        plt.show()
        
#Plot_Tradable_Z()
        
        
        
# Select randomly chosen pair from the tradable stock and visualize bid and ask prices, bid and ask volumes, and the z-values
import random

# Choose random stock
random_pair = random.choice(stock_pairs)

# Create a plot showing the bid and ask prices of a randomly chosen stock
def Plot_RandomPair_BidAskPrices():
    """
    This function plots the bid and ask price of a randomly chosen tradable pair.
    """
    plt.figure(figsize=(20,5))
    plt.title('Bid and ask prices of stock pair {} and {}'.format(random_pair[0], random_pair[1]))
    
    plt.plot(tradable_pairs_data[random_pair[0], 'AskPrice'].iloc[:100], 'r')
    plt.plot(tradable_pairs_data[random_pair[0], 'BidPrice'].iloc[:100], 'm')
    plt.xlabel('Time')
    plt.ylabel('Price stock {}'.format(random_pair[0]))
    plt.legend(loc='lower left')
    
    plt.twinx()
    plt.plot(tradable_pairs_data[random_pair[1], 'AskPrice'].iloc[:100])
    plt.plot(tradable_pairs_data[random_pair[1], 'BidPrice'].iloc[:100])
    plt.xticks([])
    plt.ylabel('Price stock {}'.format(random_pair[1]))
    plt.legend(loc='upper right')
    
    plt.show()

#Plot_RandomPair_BidAskPrices()
    
    
    
    # Create a plot showing the bid and ask volumes of a randomly chosen stock
def Plot_RandomPair_BidAskVolumes(): # Plot not really clarifying, maybe other kind of plot?
    """
    This function plots the bid and ask volumes of a randomly chosen tradable pair.
    """
    plt.figure(figsize=(20,5))
    plt.title('Bid and ask volumes of stock pair {} and {}'.format(random_pair[0],random_pair[1]))
    
    plt.plot(tradable_pairs_data[random_pair[0], 'AskVolume'].iloc[:100], 'r')
    plt.plot(tradable_pairs_data[random_pair[0], 'BidVolume'].iloc[:100], 'm')
    plt.xlabel('Time')
    plt.ylabel('Volume stock {}'.format(random_pair[0]))
    plt.legend(loc='lower left')
    
    plt.twinx()
    plt.plot(tradable_pairs_data[random_pair[1], 'AskVolume'].iloc[:100])
    plt.plot(tradable_pairs_data[random_pair[1], 'BidVolume'].iloc[:100])
    plt.xticks([])
    plt.ylabel('Volume stock {}'.format(random_pair[1]))
    plt.legend(loc='upper right')
    
    plt.show()

#Plot_RandomPair_BidAskVolumes()
    


# Create a Dataframe containing information about the error-correction term of each pair
data_error_correction_term = {'Pair': [],
                              'CountZeroCrossings': [],
                              'TradingPeriod': [],
                              'LongRunMean': [],
                              'Std': []}

for pair in stock_pairs:
    zvalue = data_zvalues[pair]
    my_array = np.array(zvalue)
    count = ((my_array[:-1] * my_array[1:]) < 0).sum()
    trading_period = 1 / count
    long_run_mean = zvalue.mean()
    std = zvalue.std()

    data_error_correction_term['Pair'].append(pair)
    data_error_correction_term['CountZeroCrossings'].append(count)
    data_error_correction_term['TradingPeriod'].append(trading_period)
    data_error_correction_term['LongRunMean'].append(round(long_run_mean, 4))
    data_error_correction_term['Std'].append(round(std, 4))

data_error_correction_term = pd.DataFrame(data_error_correction_term).set_index('Pair')

#data_error_correction_term



# Create a new column within the earlier defined DataFrame with Z-Values of all stock pairs
for pair in stock_pairs:
    stock1 = pair[0]
    stock2 = pair[1]
    
    tradable_pairs_data[stock1+stock2, 'Z-Value'] = data_zvalues[stock1,stock2]
    

# Create a Dictionary that saves all Gamma values of each pair
gamma_dictionary = {}

for pair, value in tradable_pairs_analysis.iterrows():
    gamma_dictionary[pair]= value['Gamma']
    
#gamma_dictionary
    
    
# Create a Dictionary that saves all Standard Deviation values of each pair
std_dictionary = {}

for pair, value in data_error_correction_term.iterrows():
    std_dictionary[pair]= value['Std']
    
#std_dictionary
    


"""
This is our Algorithm for finding the correct thresholds that are able to generate
 the greatest amount of profit. We find it important to not maximize the profit because
 what holds for historic data is not guaranteed to hold for future data.
 We therefore specify a limited selection of thresholds with a linspace"""


positions = {}
limit = 100

for pair in stock_pairs:
    stock1 = pair[0]
    stock2 = pair[1]
    
    gamma = gamma_dictionary[stock1,stock2]
    
    for i in np.linspace(0.05, 1.0, 10):
        threshold = i * std_dictionary[stock1,stock2]
        
        current_position_stock1 = 0 
        current_position_stock2 = 0 
        
        column_name_stock1 = stock1 + ' Pos - Thres: ' + str(threshold)
        
        BidPrice_Stock1 = tradable_pairs_data[stock1,'BidVolume'][0]
        AskPrice_Stock1 = tradable_pairs_data[stock1,'AskVolume'][0]
        BidPrice_Stock2 = tradable_pairs_data[stock2,'BidVolume'][0]
        AskPrice_Stock2 = tradable_pairs_data[stock1,'AskVolume'][0]
        
        positions[column_name_stock1] = []
        
        for time, data_at_time in tradable_pairs_data.iterrows():
            
            BidVolume_Stock1 = data_at_time[stock1, 'BidVolume']
            AskVolume_Stock1 = data_at_time[stock1, 'AskVolume']
            BidVolume_Stock2 = data_at_time[stock2, 'BidVolume']
            AskVolume_Stock2 = data_at_time[stock2, 'AskVolume']
            
            zvalue = data_at_time[stock1+stock2,'Z-Value']

            # If the zvalues of (BB,DD) are high the spread diverges, i.e. sell BB (=stock1=y) and buy DD (=stock2=x)
            if zvalue >= threshold:
                hedge_ratio = gamma * (BidPrice_Stock1 / AskPrice_Stock2)
                
                if hedge_ratio >= 1:

                    max_order_stock1 = current_position_stock1 + limit
                    max_order_stock2 = max_order_stock1 / hedge_ratio

                    trade = np.floor(min((BidVolume_Stock1 / hedge_ratio), AskVolume_Stock2, max_order_stock1, max_order_stock2))

                    positions[column_name_stock1].append((- trade * hedge_ratio) + current_position_stock1)

                    current_position_stock1 = ((- trade * hedge_ratio) + current_position_stock1)
                
                elif hedge_ratio < 1:

                    max_order_stock1 = current_position_stock1 + limit
                    max_order_stock2 = max_order_stock1 * hedge_ratio

                    trade = np.floor(min((BidVolume_Stock1 * hedge_ratio), AskVolume_Stock2, max_order_stock1, max_order_stock2))

                    positions[column_name_stock1].append((- trade / hedge_ratio) + current_position_stock1)

                    current_position_stock1 = ((- trade / hedge_ratio) + current_position_stock1)

            elif zvalue <= -threshold:
                hedge_ratio = gamma * (AskPrice_Stock1 / BidPrice_Stock2)
                
                if hedge_ratio >= 1:

                    max_order_stock1 = abs(current_position_stock1 - limit)
                    max_order_stock2 = max_order_stock1 / hedge_ratio

                    trade = np.floor(min((AskVolume_Stock1 / hedge_ratio), BidVolume_Stock2, max_order_stock1, max_order_stock2))

                    positions[column_name_stock1].append((+ trade * hedge_ratio) + current_position_stock1)

                    current_position_stock1 = (+ trade * hedge_ratio) + current_position_stock1

                elif hedge_ratio < 1:
                    
                    max_order_stock1 = abs(current_position_stock1 - limit)
                    max_order_stock2 = max_order_stock1 * hedge_ratio

                    trade = np.floor(min((AskVolume_Stock1 * hedge_ratio), BidVolume_Stock2, max_order_stock1, max_order_stock2))

                    positions[column_name_stock1].append((+ trade / hedge_ratio) + current_position_stock1)

                    current_position_stock1 = (+ trade / hedge_ratio) + current_position_stock1  
                
                BidPrice_Stock1 = data_at_time[stock1, 'BidPrice']
                AskPrice_Stock1 = data_at_time[stock1, 'AskPrice']
                BidPrice_Stock2 = data_at_time[stock2, 'BidPrice']
                AskPrice_Stock2 = data_at_time[stock2, 'AskPrice']

            else:
                    positions[column_name_stock1].append(current_position_stock1)
        
        column_name_stock2 = stock2 + ' Pos - Thres: ' + str(threshold)
        
        if hedge_ratio >= 1:
            positions[column_name_stock2] = positions[column_name_stock1] / hedge_ratio * -1
        
        elif hedge_ratio < 1:
            positions[column_name_stock2] = positions[column_name_stock1] / (1 / hedge_ratio) * -1


# Create a seperate dataframe (to keep the original dataframe intact) with rounding
# Also insert the timestamp, as found in the tradeable_pairs_data DataFrame
positions_final = np.ceil(pd.DataFrame(positions))
positions_final['Timestamp'] = tradable_pairs_data.index
positions_final = positions_final.set_index('Timestamp')



# The difference between the positions
positions_diff = positions_final.diff()[1:]

# Positions_diff first rows
#positions_diff.head()

# OPTIONAL to Excel to Save the Amount of Trades
# positions_diff[(positions_diff != 0)].count().to_excel('Thresholds.xlsx')



"""
This method is used to value our last position by the correct market value.
 This to ensure that in the time between our last trade and the last timestamp
 does not hold any secrets (a market crash for example) that are not calculated in the PnL. 
 One could say the profit is for example €50.000 while it is actually far
 lower because our positions are worth next to nothing due to a market crash."""
 
positions_diff[-1:] = -positions_final[-1:]
 
 
# To determine which threshold is the most profitable, we determine the PnL of each combination of pair and threshold

pnl_dataframe = pd.DataFrame()

for pair in stock_pairs:
    stock1 = pair[0]
    stock2 = pair[1]

    Stock1_AskPrice = tradable_pairs_data[stock1, 'AskPrice'][1:]
    Stock1_BidPrice = tradable_pairs_data[stock1, 'BidPrice'][1:]
    Stock2_AskPrice = tradable_pairs_data[stock2, 'AskPrice'][1:]
    Stock2_BidPrice = tradable_pairs_data[stock2, 'BidPrice'][1:]

    for i in np.linspace(0.05, 1.0, 10):
        threshold = i * std_dictionary[stock1, stock2]

        column_name_1 = stock1 + ' Pos - Thres: ' + str(threshold)
        column_name_2 = stock2 + ' Pos - Thres: ' + str(threshold)

        pnl_dataframe[stock1 + str(threshold)] = np.where(positions_diff[column_name_1] > 0,
                                                          positions_diff[column_name_1] * -Stock1_BidPrice, positions_diff[column_name_1] * -Stock1_AskPrice)
        pnl_dataframe[stock2 + str(threshold)] = np.where(positions_diff[column_name_2] > 0,
                                                          positions_diff[column_name_2] * -Stock2_BidPrice, positions_diff[column_name_2] * -Stock2_AskPrice)

#pnl_dataframe.head()
        
        
        
        
# Create Columns for the pnl_threshold dataframe
pairs = []
thresholds = []

for pair in stock_pairs:
    stock1 = pair[0]
    stock2 = pair[1]

    for i in np.linspace(0.05, 1.0, 10):
        threshold = i * std_dictionary[stock1, stock2]
        pair = stock1, stock2
        pairs.append(pair)
        thresholds.append(threshold)
        



# Include columns and append PnLs
pnl_threshold = {'Pairs' : pairs,
                 'Thresholds': thresholds,
                 'PnLs' : []}

for pair in stock_pairs:
    stock1 = pair[0]
    stock2 = pair[1]
    
    for i in np.linspace(0.05, 1.0, 10):
        threshold = i * std_dictionary[stock1,stock2]
        pnl_threshold['PnLs'].append(pnl_dataframe[stock1 + str(threshold)].sum() + pnl_dataframe[stock2 + str(threshold)].sum())
        
pnl_threshold = pd.DataFrame(pnl_threshold)
pnl_threshold = pnl_threshold.set_index('Pairs')
# pnl_threshold.to_excel('Thresholds.xlsx')


# Find Highest PnLs
highest_pnls = pnl_threshold.groupby(by='Pairs').agg({'PnLs' : max})
highest_pnls.sort_values('PnLs', ascending=False)


# Plot error-correction term (z-value) to observe what the spread looks like (see slide for comparison plot cointegrated pair)
def Plot_Thresholds(stock1, stock2):
    zvalue = tradable_pairs_data[stock1+stock2,'Z-Value']
    plt.figure(figsize=(20,15))
    plt.xticks([])
    plt.title('Error-correction term stock pair ' + stock1 + ' and ' + stock2)
    zvalue.plot(alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    xmin = 0
    xmax = len(zvalue)
    
    # Boundries chosen to give an approximate good fit
    plt.hlines(pnl_threshold['Thresholds'][10:20], xmin, xmax, 'g')  
    plt.hlines(-pnl_threshold['Thresholds'][10:20], xmin, xmax, 'r')
    
    plt.legend(['Z-Value', 'Positive Threshold', 'Negative Threshold'])
    plt.show()
        
#Plot_Thresholds('BB','JJ')
    



# Create a Plot that displays the Profitability of the Thresholds

def profitability_of_the_thresholds(stock1, stock2):
    pnl_threshold[(pnl_threshold.index == (stock1, stock2))].plot(x='Thresholds', y='PnLs', figsize=(10,10))
    plt.title('Profitability of the Thresholds for ' + stock1 + ' and ' + stock2)
    plt.xlabel('Amount of Sigma away from the Mean')
    plt.ylabel('Profits and Losses')
    plt.legend(['Profit and Losses'])
    plt.grid()

#profitability_of_the_thresholds('BB','JJ')
    
    
    
    
    
    
    
    
    
    
################################## ALGORITHM STRATEGY SECTION ############################### 
"""Algorithm Strategy 1
Making use of the previous analysis to determine which pairs
 should be traded. Based on that the algorithm, with slight modifications, 
is ran again to calculate the final profits."""


# Determine the treshold, manually chosen based on pnl_threshold and ensuring no overlap.
threshold_dictionary = {('BB', 'JJ'): 0.000220,
                        ('FF', 'MM'): 0.000155,
                        ('DD', 'HH'): 0.000485,
                        ('AA', 'II'): 0.001070}

#threshold_dictionary



# Selection of the final pairs for this trading strategy
stock_pairs_final = [('BB', 'JJ'),
                     ('FF', 'MM'),
                     ('DD', 'HH'),
                     ('AA', 'II')]

#stock_pairs_final



"""
This algorithm is a slight modification as the previous one used.
 In this algorithm we incorporate the chosen pairs,
 with the corresponding tresholds, to determine the most optimal positions."""
 
positions_strategy_1 = {}
limit = 100

for pair in stock_pairs_final:
    stock1 = pair[0]
    stock2 = pair[1]
    
    gamma = gamma_dictionary[stock1,stock2]
    
    threshold = threshold_dictionary[stock1,stock2]
        
    current_position_stock1 = 0 
    current_position_stock2 = 0 
        
    positions_strategy_1[stock1] = []

    for time, data_at_time in tradable_pairs_data.iterrows():

        BidPrice_Stock1 = data_at_time[stock1, 'BidPrice']
        AskPrice_Stock1 = data_at_time[stock1, 'AskPrice']
        BidPrice_Stock2 = data_at_time[stock2, 'BidPrice']
        AskPrice_Stock2 = data_at_time[stock2, 'AskPrice']

        BidVolume_Stock1 = data_at_time[stock1, 'BidVolume']
        AskVolume_Stock1 = data_at_time[stock1, 'AskVolume']
        BidVolume_Stock2 = data_at_time[stock2, 'BidVolume']
        AskVolume_Stock2 = data_at_time[stock2, 'AskVolume']

        zvalue = data_at_time[stock1+stock2,'Z-Value']

        if zvalue >= threshold:
            hedge_ratio = gamma * (BidPrice_Stock1 / AskPrice_Stock2)
                
            if hedge_ratio >= 1:

                max_order_stock1 = current_position_stock1 + limit
                max_order_stock2 = max_order_stock1 / hedge_ratio

                trade = np.floor(min((BidVolume_Stock1 / hedge_ratio), AskVolume_Stock2, max_order_stock1, max_order_stock2))

                positions_strategy_1[stock1].append((- trade * hedge_ratio) + current_position_stock1)

                current_position_stock1 = ((- trade * hedge_ratio) + current_position_stock1)
                
            elif hedge_ratio < 1:

                max_order_stock1 = current_position_stock1 + limit
                max_order_stock2 = max_order_stock1 * hedge_ratio

                trade = np.floor(min((BidVolume_Stock1 * hedge_ratio), AskVolume_Stock2, max_order_stock1, max_order_stock2))

                positions_strategy_1[stock1].append((- trade / hedge_ratio) + current_position_stock1)

                current_position_stock1 = ((- trade / hedge_ratio) + current_position_stock1)

        elif zvalue <= -threshold:
            hedge_ratio = gamma * (AskPrice_Stock1 / BidPrice_Stock2)
                
            if hedge_ratio >= 1:

                max_order_stock1 = abs(current_position_stock1 - limit)
                max_order_stock2 = max_order_stock1 / hedge_ratio

                trade = np.floor(min((AskVolume_Stock1 / hedge_ratio), BidVolume_Stock2, max_order_stock1, max_order_stock2))

                positions_strategy_1[stock1].append((+ trade * hedge_ratio) + current_position_stock1)

                current_position_stock1 = (+ trade * hedge_ratio) + current_position_stock1

            elif hedge_ratio < 1:
                    
                max_order_stock1 = abs(current_position_stock1 - limit)
                max_order_stock2 = max_order_stock1 * hedge_ratio

                trade = np.floor(min((AskVolume_Stock1 * hedge_ratio), BidVolume_Stock2, max_order_stock1, max_order_stock2))

                positions_strategy_1[stock1].append((+ trade / hedge_ratio) + current_position_stock1)

                current_position_stock1 = (+ trade / hedge_ratio) + current_position_stock1   

        else:

                positions_strategy_1[stock1].append(current_position_stock1)
        
    if hedge_ratio >= 1:
        positions_strategy_1[stock2] = positions_strategy_1[stock1] / hedge_ratio * -1
        
    elif hedge_ratio < 1:
        positions_strategy_1[stock2] = positions_strategy_1[stock1] / (1 / hedge_ratio) * -1



# Set Ceiling (to prevent positions with not enough volume available) as well as define the timestamp
positions_strategy_1 = np.ceil(pd.DataFrame(positions_strategy_1)) # np.ceil This function returns the ceil value of the input array elements
positions_strategy_1['Timestamp'] = tradable_pairs_data.index
positions_strategy_1 = positions_strategy_1.set_index('Timestamp')


# The difference between the positions
positions_diff_strategy_1 = positions_strategy_1.diff()[1:]

# # Positions_diff first rows
# positions_diff_strategy_1.head()




#Used as mentioned earlier.
positions_diff_strategy_1[-1:] = -positions_strategy_1[-1:]

# Show Positions over Time
for pairs in stock_pairs_final:
    stock1 = pairs[0]
    stock2 = pairs[1]
    
    plt.figure(figsize=(20,5))
    
    positions_strategy_1[stock1].plot()
    positions_strategy_1[stock2].plot()
    
    plt.title('Positions over Time for ' + stock1 + ' and ' + stock2)
    plt.legend(["Position in " + stock1,"Position in " + stock2], loc='lower right')
    
    plt.show()
    



pnl_dataframe_strategy_1 = pd.DataFrame()

for pair in stock_pairs_final:
    stock1 = pair[0]
    stock2 = pair[1]
    
    Stock1_AskPrice = tradable_pairs_data[stock1,'AskPrice'][1:]
    Stock1_BidPrice = tradable_pairs_data[stock1,'BidPrice'][1:]
    Stock2_AskPrice = tradable_pairs_data[stock2,'AskPrice'][1:]
    Stock2_BidPrice = tradable_pairs_data[stock2,'BidPrice'][1:]

    pnl_dataframe_strategy_1[stock1] = np.where(positions_diff_strategy_1[stock1] > 0, positions_diff_strategy_1[stock1] * -Stock1_BidPrice, positions_diff_strategy_1[stock1] * -Stock1_AskPrice)
    pnl_dataframe_strategy_1[stock2] = np.where(positions_diff_strategy_1[stock2] > 0, positions_diff_strategy_1[stock2] * -Stock2_BidPrice, positions_diff_strategy_1[stock2] * -Stock2_AskPrice)

#print("The total profit is: €",round(pnl_dataframe_strategy_1.sum().sum()))
    
    
    
    
    
pnl_dataframe_strategy_1['Timestamp'] = tradable_pairs_data.index[1:]
pnl_dataframe_strategy_1 = pnl_dataframe_strategy_1.set_index('Timestamp')

pnl_dataframe_strategy_1['PnL'] = pnl_dataframe_strategy_1.sum(axis=1)
pnl_dataframe_strategy_1['Cum PnL'] = pnl_dataframe_strategy_1['PnL'].cumsum()

for pair in stock_pairs_final:
    stock1 = pair[0]
    stock2 = pair[1]

    pnl_dataframe_strategy_1[stock1+stock2 + ' PnL'] = pnl_dataframe_strategy_1[stock1] + pnl_dataframe_strategy_1[stock2]
    pnl_dataframe_strategy_1[stock1+stock2 + ' Cum PnL'] = pnl_dataframe_strategy_1[stock1+stock2 + ' PnL'].cumsum()

#pnl_dataframe_strategy_1.tail()




# All Pairs's PnL

for pair in stock_pairs_final:
    stock1 = pair[0]
    stock2 = pair[1]
    
    pnl_dataframe_strategy_1[stock1+stock2 + ' Cum PnL'].plot(figsize=(10,10))
    plt.title('Cumulative PnL of ' + stock1 + ' and ' + stock2)
    plt.ylabel('Profit and Loss')
    plt.xlabel("")
    plt.grid()
    plt.xticks(rotation=20)
    plt.show()
    
    
    
# All Pairs's PnLs (including total) in one graph

pnl_dataframe_strategy_1['Cum PnL'].plot()

for pair in stock_pairs_final:
    stock1 = pair[0]
    stock2 = pair[1]
    
    pnl_dataframe_strategy_1[stock1+stock2 + ' Cum PnL'].plot(figsize=(10,10))
    plt.legend(['Cum PnL', 'BB and JJ Cum PnL', 'FF and MM Cum PnL', 'DD and HH Cum PnL','AA and II Cum PnL'])
    plt.title('Cumulative PnLs of the Trading Strategy')
    plt.ylabel('Profit and Loss')
    plt.xlabel("")
    plt.grid()
    plt.xticks(rotation=20)












######################TEST SECTION################################################


def test_HFT_Data_Sort_TC1(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(read_data('Data/Pairs Trading.csv'), file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("this will be recorded \n"
             'ReqID :HFT_Data_Sort_SF012 \n'
             'Test Suite:TS-32')
        

def test_Ask_and_Bid_Price_TC02(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(bid_ask_price_plot("CC", "MM"), file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("this will be recorded \n"
             'ReqID :HFT_Bid_and_Ask_Price_SF013 \n'
             'Test Suite:TS-33')
        
def test_Ask_and_Bid_Volume_TC03(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(bid_ask_volume_plot("CC", "MM"), file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("this will be recorded \n"
             'ReqID :HFT_Bid_and_Ask_Volume_SF014 \n'
             'Test Suite:TS-34')
    
def test_Market_Data_MidPrice_TC04(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(market_data, file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("\n this will be recorded \n"
             'ReqID :HFT_Market_Data_MidPrice_SF015 \n'
             'Test Suite:TS-35')
        

def test_Market_Data_MidPrice_Check_TC05(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(mid_price_check('MM'), file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("\n this will be recorded \n"
             'ReqID :HFT_Market_Data_MidPrice_Check_SF016 \n'
             'Test Suite:TS-36')
        
        
def test_Statistical_Parameters_TC06(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(data_analysis, file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("\n this will be recorded \n"
             'ReqID :HFT_Statistical_Parameters_SF017 \n'
             'Test Suite:TS-37')
        
def test_Tradable_Pairs_Analysis_TC07(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(tradable_pairs_analysis, file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("\n this will be recorded \n"
             'ReqID :HFT_Tradable_Pairs_Analysis_SF018 \n'
             'Test Suite:TS-38')
        
        
        
        
def test_Tradable_Pairs_Data_ALGO_TC18(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(tradable_pairs_data, file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("\n this will be recorded \n"
             'ReqID :HFT_Tradable_Pairs_Data_ALGO_SF029 \n'
             'Test Suite:TS-39')
        
        
def test_Pairs_Error_Correction_Term_ALGO_TC20(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(data_error_correction_term, file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("\n this will be recorded \n"
             'ReqID :HFT_Pairs_Error_correction_Term_ALGO_SF030 \n'
             'Test Suite:TS-40')
        

def test_Positional_diff_Threshold_ALGO_TC21(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(positions_diff, file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("\n this will be recorded \n"
             'ReqID :HFT_Positional_Diff_Threshold_ALGO_SF031 \n'
             'Test Suite:TS-41')
        
def test_Profitability_Threshold_ALGO_TC22(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(pnl_dataframe, file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("\n this will be recorded \n"
             'ReqID :HFT_Profitability_Threshold_ALGO_SF032 \n'
             'Test Suite:TS-42')
        
        
def test_Manual_Threshold_AST_TC23(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(threshold_dictionary, file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("\n this will be recorded \n"
             'ReqID :HFT_Manual_Threshold_AST_SF033 \n'
             'Test Suite:TS-43')
        
        
def test_Position_Diff__AST_TC24(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(positions_diff_strategy_1, file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("\n this will be recorded \n"
             'ReqID :HFT_Position_Diff_AST_SF034 \n'
             'Test Suite:TS-44')
        
        

        
def test_Pnl__AST_TC25(regtest):
    #ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # one way to record output:
    print(pnl_dataframe_strategy_1, file=regtest)

    # alternative method to record output:
    regtest.write("done")

    # or using a context manager:
    with regtest:
        print("\n this will be recorded \n"
             'ReqID :HFT_Profit_and_loss_AST_SF035 \n'
             'Test Suite:TS-45')