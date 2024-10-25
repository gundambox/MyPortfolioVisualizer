# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# # Step 1: Download Historical Data for ^TWII
# # Retrieve historical data for ^TWII starting from 2004/01/01 to the present
# # Discard non-trading days (twiiHistory)
# twii = yf.Ticker("^TWII")
# twiiHistory = twii.history(period="max")
# twiiHistory = twiiHistory.dropna()
# twiiHistory.to_csv('twiiHistory.csv')
# # Step 2: Download Historical Data for 00631L.TW
# # Retrieve historical data for 00631L.TW starting from 2014/12/01 to the present
# # Discard non-trading days (etf00631LHistory)
# etf00631L = yf.Ticker("00631L.TW")
# etf00631LHistory = etf00631L.history(start="2014-12-01")
# etf00631LHistory = etf00631LHistory.dropna()
# etf00631LHistory.to_csv('etf00631LHistory.csv')

# # Read from CSV
# twiiHistory = pd.read_csv('twiiHistory.csv', index_col='Date', parse_dates=True)
# etf00631LHistory = pd.read_csv('etf00631LHistory.csv', index_col='Date', parse_dates=True)

# # Step 3: Simulate 00631L.TW's Returns Using ^TWII Data
# # Use the ^TWII percetage change data from 2004/01/01 to 2014/11/30 to simulate the returns of 00631L.TW
# #   Calculate the average and standard deviation of 00631L.TW's percetage change (etf00631LMean, etf00631LStd)
# #   Calculate the average and standard deviation of ^TWII's percetage change (twiiMean, twiiStd)
# #   Simulate the percetage change of 00631L.TW by multiplying ^TWII's percetage change by 2 and adding random noise based on 00631L.TW's historical returns
# #   Combine the historical percetage change of 00631L.TW with the simulated returns to form the complete data set (etf00631LFinal)
# #   Change column name "Close" to "Percentage" and save the simulated data set to a CSV file
# etf00631LPctChange = etf00631LHistory['Close'].pct_change()
# etf00631LMean = etf00631LPctChange.mean()
# etf00631LStd = etf00631LPctChange.std()
# # print('etf00631LMean:', etf00631LMean)
# # print('etf00631LStd:', etf00631LStd)
# twiiPctChange = twiiHistory['Close'].pct_change()
# twiiMean = twiiPctChange.mean()
# twiiStd = twiiPctChange.std()
# # print('twiiMean:', twiiMean)
# # print('twiiStd:', twiiStd)

# np.random.seed(0)
# simulateLength = len(twiiPctChange) - len(etf00631LPctChange)
# simulated_etf00631LPctChange = twiiPctChange[:simulateLength] * 2

# # concat simulated_etf00631LPctChange and etf00631LPctChange and name the columen as 'Percentage'
# etf00631LFinal = pd.concat([simulated_etf00631LPctChange, etf00631LPctChange])
# etf00631LFinal = pd.DataFrame(etf00631LFinal, columns=['Close'])
# etf00631LFinal = etf00631LFinal.rename(columns={'Close': 'Percentage'})
# etf00631LFinal = etf00631LFinal.dropna()

# twiiFinal = twiiPctChange.copy()
# twiiFinal = pd.DataFrame(twiiFinal, index=twiiHistory.index, columns=['Close'])
# twiiFinal = twiiFinal.rename(columns={'Close': 'Percentage'})
# twiiFinal = twiiFinal.dropna()

# # # # Step 4: Find the Intersection of Both Data Sets
# # # # Retain only the trading days present in both data sets
# # # # This ensures that we have a common set of dates for analysis
# # # # Save the intersected data sets to CSV files
# common_dates = twiiFinal.index.intersection(etf00631LFinal.index)
# twiiFinal = twiiFinal.loc[common_dates]
# etf00631LFinal = etf00631LFinal.loc[common_dates]

# twiiFinal.to_csv('twiiFinal.csv')
# etf00631LFinal.to_csv('etf00631LFinal.csv')

twiiFinal = pd.read_csv('twiiFinal.csv', index_col='Date', parse_dates=True)
etf00631LFinal = pd.read_csv('etf00631LFinal.csv', index_col='Date', parse_dates=True)

# Change index format to python date
twiiFinal.index = pd.to_datetime(twiiFinal.index)
etf00631LFinal.index = pd.to_datetime(etf00631LFinal.index)

# Ask for user input to get the start date
start_date = input("Enter the start date (YYYY-MM-DD): ")
# Filter the data based on the start date, if the start date is not in the data set, filter out the dates before the start date
twiiFinal = twiiFinal.loc[(twiiFinal.index >= start_date)]
etf00631LFinal = etf00631LFinal.loc[(etf00631LFinal.index >= start_date)]

# Step 5: Investment Simulation Starting with 1,000,000 TWD
# Assume an initial investment of 1,000,000 TWD on the first trading day after 2004/01/01
# Simulate asset changes using the following strategies:
#   Strategy 1: Reallocate all assets to ^TWII on the first trading day of each year
#   Strategy 2: Reallocate all assets to 00631L.TW on the first trading day of each year
#   Strategy 3: Allocate 40% of assets to 00631L.TW and hold the remainder in cash on the first trading day of each year
#   Strategy 4: Allocate 50% of assets to 00631L.TW and hold the remainder in cash on the first trading day of each year
#   Strategy 5: Allocate 60% of assets to 00631L.TW and hold the remainder in cash on the first trading day of each year

initial_investment = 1000000
strategies = {
    "Strategy 1": {"twii": 1.0, "etf": 0.0, "cash": 0.0},
    "Strategy 2": {"twii": 0.0, "etf": 1.0, "cash": 0.0},
    "Strategy 3": {"twii": 0.0, "etf": 0.4, "cash": 0.6},
    "Strategy 4": {"twii": 0.0, "etf": 0.5, "cash": 0.5},
    "Strategy 5": {"twii": 0.0, "etf": 0.6, "cash": 0.4},
}

# Initialize the asset changes for each strategy
asset_changes = { strategy: {asset: [int(initial_investment*ratio)] for asset, ratio in allocation.items()} for strategy, allocation in strategies.items() }

# Get common dates
common_dates = twiiFinal.index.intersection(etf00631LFinal.index)

# Allocate assets based on each strategy first trading day of each year
last_year = twiiFinal.index[0].year
for date in common_dates:
    # Check if the year has changed
    if date.year != last_year:
        for strategy, allocation in strategies.items():
            # reallocate assets based on the strategy
            total_value = asset_changes[strategy]["twii"][-1] + asset_changes[strategy]["etf"][-1] + asset_changes[strategy]["cash"][-1]
            twii_allocation = total_value * allocation["twii"]
            etf_allocation = total_value * allocation["etf"]
            cash_allocation = total_value * allocation["cash"]
            # Update asset changes, because we are using currency, we round to integer
            asset_changes[strategy]["twii"].append(int(twii_allocation * (1 + twii_change)))
            asset_changes[strategy]["etf"].append(int(etf_allocation * (1 + etf_change)))
            asset_changes[strategy]["cash"].append(int(cash_allocation))

        last_year = date.year

    else:
        for strategy, allocation in strategies.items():
            twii_change = twiiFinal.loc[date, "Percentage"]
            etf_change = etf00631LFinal.loc[date, "Percentage"]

            asset_changes[strategy]["twii"].append(int(asset_changes[strategy]["twii"][-1] * (1 + twii_change)))
            asset_changes[strategy]["etf"].append(int(asset_changes[strategy]["etf"][-1] * (1 + etf_change)))
            asset_changes[strategy]["cash"].append(int(asset_changes[strategy]["cash"][-1]))

# Save changes to CSV
for strategy, changes in asset_changes.items():
    # Ensure the lengths match
    min_length = min(len(common_dates), len(changes["twii"]))
    df = pd.DataFrame({asset: values[:min_length] for asset, values in changes.items()}, index=common_dates[:min_length])
    df.to_csv(f'{strategy}.csv')


# Step 6: Calculate Metrics
# For each strategy, calculate the annualized return, standard deviation, Sharpe ratio, maximum drawdown, maximum consecutive gain days, and maximum consecutive loss days
# Print the metrics for each strategy
for strategy, allocation in strategies.items():
    print(f"Strategy: {strategy}")
    twii_values = asset_changes[strategy]["twii"]
    etf_values = asset_changes[strategy]["etf"]
    cash_values = asset_changes[strategy]["cash"]
    total_values = [twii + etf + cash for twii, etf, cash in zip(twii_values, etf_values, cash_values)]
    total_returns = pd.Series(total_values)
    cumulative_return = (total_returns.iloc[-1] - total_returns.iloc[0]) / total_returns.iloc[0]
    annualized_return = (total_returns.iloc[-1] / total_returns.iloc[0]) ** (252 / len(total_returns)) - 1
    annualized_std = total_returns.pct_change().std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_std
    max_drawdown = (total_returns / total_returns.cummax() - 1).min()

    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Standard Deviation: {annualized_std:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print("="*40)

# Step 7: Plot Asset Changes
# Create a visualization of asset changes over time for each strategy
# Save the plot to a file

plt.figure(figsize=(14, 7))

for strategy, allocation in strategies.items():
    twii_values = asset_changes[strategy]["twii"]
    etf_values = asset_changes[strategy]["etf"]
    cash_values = asset_changes[strategy]["cash"]
    total_values = [twii + etf + cash for twii, etf, cash in zip(twii_values, etf_values, cash_values)]
    plt.plot(common_dates, total_values[:len(common_dates)], label=strategy)

plt.title('Asset Changes Over Time for Each Strategy')
plt.xlabel('Date')
plt.ylabel('Total Asset Value (TWD)')
plt.legend()
plt.grid(True)
plt.savefig('asset_changes.png')
plt.show()
