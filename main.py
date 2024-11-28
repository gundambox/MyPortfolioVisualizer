# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import math

ticket_list = ["^TWII", "00631L.TW", "QQQ", "QLD", "SPY", "SSO", "BND"]
CASH_TWD_ANNUAL_RATE = 0.015
CASH_USD_ANNUAL_RATE = 0.03
INFLATION_RATE = 0.023


def download_data(path):
    # Download the data from Yahoo Finance
    history = yf.download(ticket_list, interval="1mo", ignore_tz=True, auto_adjust=True, actions=True)
    history.to_csv(f'Origin_{path}')
    history = history['Close']
    history.to_csv(path)
    return history


def load_data(path):
    if os.path.exists(path):
        history = pd.read_csv(path)
        # # Convert column name 'Ticker' to 'Date' and set it as index
        # last_trading_day = pd.to_datetime(
        #     history['Date']).max().to_pydatetime().date()
        # # if last trading day is not today, download the latest data
        # if last_trading_day < (datetime.datetime.today() + datetime.timedelta(days=-1)).date():
        #     print("Download the latest data")
        #     history = download_data(path)
    else:
        history = download_data(path)

    return history


def extend_leverage_data(origin_data, leverage_data):
    origin_pct_change_data = origin_data['Price'].pct_change(fill_method=None)
    leverage_pct_change_data = leverage_data['Price'].pct_change(
        fill_method=None)
    np.random.seed(0)

    # Find the first valid index of the origin_pct_change_data
    origin_pct_change_data_idx = pd.Series(
        origin_pct_change_data).first_valid_index()
    # Find the first valid index of the leverage_pct_change_data
    leverage_pct_change_data_idx = pd.Series(
        leverage_pct_change_data).first_valid_index()

    # pick the range from origin_pct_change_data_idx to leverage_pct_change_data_idx and multiply 2 and reassign back to leverage_pct_change_data
    leverage_final = pd.concat([origin_pct_change_data[:origin_pct_change_data_idx],
                                origin_pct_change_data[origin_pct_change_data_idx:
                                                       leverage_pct_change_data_idx] * 2,
                                leverage_pct_change_data[leverage_pct_change_data_idx:]])
    leverage_final = pd.DataFrame(
        {'Date': origin_data['Date'], 'Percentage': leverage_final})
    return leverage_final


def filter_data_by_start_date(data, start_date):
    return data[data['Date'] >= start_date]


def simulatedPortfolioChange(start_date, tradingDate, initial_investment, annualCashFlow, twiiFinal, etf00631LFinal, etfQQQFinal, etfQLDFinal, etfSPYFinal, etfSSOFinal, etfBNDFinal):
    # Filter the data by start date

    strategies = {
        # 100% TWII
        "100% TWII": {
            "cash-TWD": 0.0,
            "cash-USD": 0.0,
            "stock-twii": 1.0,
            "stock-00631L": 0.0,
            "stock-QQQ": 0.0,
            "stock-QLD": 0.0,
            "stock-SPY": 0.0,
            "stock-SSO": 0.0,
            "stock-BND": 0.0
        },
        # 100% QLD
        "100% QLD": {
            "cash-TWD": 0.0,
            "cash-USD": 0.0,
            "stock-twii": 0.0,
            "stock-00631L": 0.0,
            "stock-QQQ": 0.0,
            "stock-QLD": 1.0,
            "stock-SPY": 0.0,
            "stock-SSO": 0.0,
            "stock-BND": 0.0
        },
        # 100% SSO
        "100% SSO": {
            "cash-TWD": 0.0,
            "cash-USD": 0.0,
            "stock-twii": 0.0,
            "stock-00631L": 0.0,
            "stock-QQQ": 0.0,
            "stock-QLD": 0.0,
            "stock-SPY": 0.0,
            "stock-SSO": 1.0,
            "stock-BND": 0.0
        },
        # 100% SPY
        "100% SPY": {
            "cash-TWD": 0.0,
            "cash-USD": 0.0,
            "stock-twii": 0.0,
            "stock-00631L": 0.0,
            "stock-QQQ": 0.0,
            "stock-QLD": 0.0,
            "stock-SPY": 1.0,
            "stock-SSO": 0.0,
            "stock-BND": 0.0
        },
        # 100% QQQ
        "100% QQQ": {
            "cash-TWD": 0.0,
            "cash-USD": 0.0,
            "stock-twii": 0.0,
            "stock-00631L": 0.0,
            "stock-QQQ": 1.0,
            "stock-QLD": 0.0,
            "stock-SPY": 0.0,
            "stock-SSO": 0.0,
            "stock-BND": 0.0
        },
    }

    # Initialize the asset changes for each strategy
    asset_changes = {strategy: {asset: [int(initial_investment*ratio)] for asset,
                                ratio in allocation.items()} for strategy, allocation in strategies.items()}

    # Allocate assets based on each strategy first trading day of each year
    last_year = datetime.datetime.strptime(start_date, "%Y-%m-%d").year - 1

    for date in tradingDate:
        # Check if the year has changed
        if pd.to_datetime(date).year != last_year:
            # Rebalance the portfolio
            print(
                f"First trading day of the year {pd.to_datetime(date).year}, rebalance the portfolio")
            for strategy, allocation in strategies.items():
                # reallocate assets based on the strategy
                asset_cashTWD = asset_changes[strategy]["cash-TWD"][-1]
                asset_cashUSD = asset_changes[strategy]["cash-USD"][-1]
                asset_stockWwii = asset_changes[strategy]["stock-twii"][-1]
                asset_stock00631L = asset_changes[strategy]["stock-00631L"][-1]
                asset_stockQQQ = asset_changes[strategy]["stock-QQQ"][-1]
                asset_stockQLD = asset_changes[strategy]["stock-QLD"][-1]
                asset_stockSPY = asset_changes[strategy]["stock-SPY"][-1]
                asset_stockSSO = asset_changes[strategy]["stock-SSO"][-1]
                asset_stockBND = asset_changes[strategy]["stock-BND"][-1]

                total_value = asset_cashTWD + asset_cashUSD + asset_stockWwii + asset_stock00631L + \
                    asset_stockQQQ + asset_stockQLD + asset_stockSPY + \
                    asset_stockSSO + asset_stockBND + annualCashFlow

                cashTWD_allocation = total_value * allocation["cash-TWD"]
                cashUSD_allocation = total_value * allocation["cash-USD"]
                twii_allocation = total_value * allocation["stock-twii"]
                etf_allocation = total_value * allocation["stock-00631L"]
                QQQ_allocation = total_value * allocation["stock-QQQ"]
                QLD_allocation = total_value * allocation["stock-QLD"]
                SPY_allocation = total_value * allocation["stock-SPY"]
                SSO_allocation = total_value * allocation["stock-SSO"]
                BND_allocation = total_value * allocation["stock-BND"]

                # Update last record asset changes, because we are using currency, we round to integer
                asset_changes[strategy]["cash-TWD"][-1] = int(
                    cashTWD_allocation)
                asset_changes[strategy]["cash-USD"][-1] = int(
                    cashUSD_allocation)
                asset_changes[strategy]["stock-twii"][-1] = int(
                    twii_allocation)
                asset_changes[strategy]["stock-00631L"][-1] = int(
                    etf_allocation)
                asset_changes[strategy]["stock-QQQ"][-1] = int(QQQ_allocation)
                asset_changes[strategy]["stock-QLD"][-1] = int(QLD_allocation)
                asset_changes[strategy]["stock-SPY"][-1] = int(SPY_allocation)
                asset_changes[strategy]["stock-SSO"][-1] = int(SSO_allocation)
                asset_changes[strategy]["stock-BND"][-1] = int(BND_allocation)
            last_year = pd.to_datetime(date).year

        for strategy, allocation in strategies.items():
            # Get the percentage change of the stock according to the date, if this date does not have trading data, assume the stock price does not change
            twii_change = twiiFinal.loc[twiiFinal['Date']
                                        == date, 'Percentage']
            asset_changes[strategy]["stock-twii"].append(
                int(asset_changes[strategy]["stock-twii"][-1] * (1 + twii_change)))

            etf00631L_change = etf00631LFinal.loc[etf00631LFinal['Date']
                                                  == date, 'Percentage']
            asset_changes[strategy]["stock-00631L"].append(
                int(asset_changes[strategy]["stock-00631L"][-1] * (1 + etf00631L_change)))

            etfQQQ_change = etfQQQFinal.loc[etfQQQFinal['Date']
                                            == date, 'Percentage']
            asset_changes[strategy]["stock-QQQ"].append(
                int(asset_changes[strategy]["stock-QQQ"][-1] * (1 + etfQQQ_change)))

            etfQLD_change = etfQLDFinal.loc[etfQLDFinal['Date']
                                            == date, 'Percentage']
            asset_changes[strategy]["stock-QLD"].append(
                int(asset_changes[strategy]["stock-QLD"][-1] * (1 + etfQLD_change)))

            etfSPY_change = etfSPYFinal.loc[etfSPYFinal['Date']
                                            == date, 'Percentage']
            asset_changes[strategy]["stock-SPY"].append(
                int(asset_changes[strategy]["stock-SPY"][-1] * (1 + etfSPY_change)))

            etfSSO_change = etfSSOFinal.loc[etfSSOFinal['Date']
                                            == date, 'Percentage']
            asset_changes[strategy]["stock-SSO"].append(
                int(asset_changes[strategy]["stock-SSO"][-1] * (1 + etfSSO_change)))
            etfBND_change = etfBNDFinal.loc[etfBNDFinal['Date']
                                            == date, 'Percentage']
            asset_changes[strategy]["stock-BND"].append(
                int(asset_changes[strategy]["stock-BND"][-1] * (1 + etfBND_change)))

            # TWD cash average fixed deposit annual interest rate is 1%
            cashTWD_monthly_rate = (1 + CASH_TWD_ANNUAL_RATE) ** (1/12) - 1
            asset_changes[strategy]["cash-TWD"].append(
                int(asset_changes[strategy]["cash-TWD"][-1] * (1 + cashTWD_monthly_rate)))
            # USD cash average fixed deposit annual interest rate is 2%
            cashUSD_monthly_rate = (1 + CASH_USD_ANNUAL_RATE) ** (1/12) - 1
            asset_changes[strategy]["cash-USD"].append(
                int(asset_changes[strategy]["cash-USD"][-1] * (1 + cashUSD_monthly_rate)))

            # Include the inflation rate
            monthly_inflation_rate = (1 + INFLATION_RATE) ** (1/12) - 1
            asset_changes[strategy]["stock-twii"][-1] = int(
                asset_changes[strategy]["stock-twii"][-1] * (1 - monthly_inflation_rate))
            asset_changes[strategy]["stock-00631L"][-1] = int(
                asset_changes[strategy]["stock-00631L"][-1] * (1 - monthly_inflation_rate))
            asset_changes[strategy]["stock-QQQ"][-1] = int(
                asset_changes[strategy]["stock-QQQ"][-1] * (1 - monthly_inflation_rate))
            asset_changes[strategy]["stock-QLD"][-1] = int(
                asset_changes[strategy]["stock-QLD"][-1] * (1 - monthly_inflation_rate))
            asset_changes[strategy]["stock-SPY"][-1] = int(
                asset_changes[strategy]["stock-SPY"][-1] * (1 - monthly_inflation_rate))
            asset_changes[strategy]["stock-SSO"][-1] = int(
                asset_changes[strategy]["stock-SSO"][-1] * (1 - monthly_inflation_rate))
            asset_changes[strategy]["stock-BND"][-1] = int(
                asset_changes[strategy]["stock-BND"][-1] * (1 - monthly_inflation_rate))
            asset_changes[strategy]["cash-TWD"][-1] = int(
                asset_changes[strategy]["cash-TWD"][-1] * (1 - monthly_inflation_rate))
            asset_changes[strategy]["cash-USD"][-1] = int(
                asset_changes[strategy]["cash-USD"][-1] * (1 - monthly_inflation_rate))

    min_length = len(tradingDate)

    # For each strategy, calculate the annualized return, standard deviation, Sharpe ratio, maximum drawdown, maximum consecutive gain days, and maximum consecutive loss days
    # Print the metrics for each strategy
    # Save report to file
    with open(f"{start_date}_report.txt", "w") as f:
        for strategy, allocation in strategies.items():
            print(f"Strategy: {strategy}")
            stockTwii_values = asset_changes[strategy]["stock-twii"]
            stock00631L_values = asset_changes[strategy]["stock-00631L"]
            stockQQQ_values = asset_changes[strategy]["stock-QQQ"]
            stockQLD_values = asset_changes[strategy]["stock-QLD"]
            stockSPY_values = asset_changes[strategy]["stock-SPY"]
            stockSSO_values = asset_changes[strategy]["stock-SSO"]
            stockBND_values = asset_changes[strategy]["stock-BND"]
            cashTWD_values = asset_changes[strategy]["cash-TWD"]
            cashUSD_values = asset_changes[strategy]["cash-USD"]

            total_values = [twii + etf + QQQ + QLD + SPY + SSO + BND + cashTWD + cashUSD for twii, etf, QQQ, QLD, SPY, SSO, BND, cashTWD, cashUSD in zip(
                stockTwii_values, stock00631L_values, stockQQQ_values, stockQLD_values, stockSPY_values, stockSSO_values, stockBND_values, cashTWD_values, cashUSD_values)]
            total_returns = pd.Series(total_values)
            cumulative_return = (
                total_returns.iloc[-1] - total_returns.iloc[0]) / total_returns.iloc[0]
            annualized_return = (
                total_returns.iloc[-1] / total_returns.iloc[0]) ** (12 / len(total_returns)) - 1
            annualized_std = total_returns.pct_change().std() * np.sqrt(12)
            max_drawdown = (total_returns / total_returns.cummax() - 1).min()

            print(f"Strategy: {strategy}")
            print(f"最終資產: {total_values[-1]}")
            print(f"總報酬: {cumulative_return:.2%}")
            print(f"年化報酬: {annualized_return:.2%}")
            print(f"年化標準差: {annualized_std:.2%}")
            print(f"最大跌幅: {max_drawdown:.2%}")
            print("="*40)

            f.write(f"Strategy: {strategy}\n")
            f.write(f"總報酬: {cumulative_return:.2%}\n")
            f.write(f"最終資產: {total_values[-1]}\n")
            f.write(f"年化報酬: {annualized_return:.2%}\n")
            f.write(f"年化標準差: {annualized_std:.2%}\n")
            f.write(f"最大跌幅: {max_drawdown:.2%}\n")
            f.write("="*40+"\n")

    # Step 7: Plot Asset Changes
    # Create a visualization of asset changes over time for each strategy
    # Save the plot to a file

    plt.figure(figsize=(14, 7))

    final_asset_values = {}
    for strategy, allocation in strategies.items():
        stockTwii_values = asset_changes[strategy]["stock-twii"][:min_length]
        stock00631L_values = asset_changes[strategy]["stock-00631L"][:min_length]
        stockQQQ_values = asset_changes[strategy]["stock-QQQ"][:min_length]
        stockQLD_values = asset_changes[strategy]["stock-QLD"][:min_length]
        stockSPY_values = asset_changes[strategy]["stock-SPY"][:min_length]
        stockSSO_values = asset_changes[strategy]["stock-SSO"][:min_length]
        stockBND_values = asset_changes[strategy]["stock-BND"][:min_length]
        cashTWD_values = asset_changes[strategy]["cash-TWD"][:min_length]
        cashUSD_values = asset_changes[strategy]["cash-USD"][:min_length]

        total_values = [twii + etf + QQQ + QLD + SPY + SSO + BND + cashTWD + cashUSD for twii, etf, QQQ, QLD, SPY, SSO, BND, cashTWD, cashUSD in zip(
            stockTwii_values, stock00631L_values, stockQQQ_values, stockQLD_values, stockSPY_values, stockSSO_values, stockBND_values, cashTWD_values, cashUSD_values)]
        final_asset_values[strategy] = total_values
        plt.plot(tradingDate, total_values, label=strategy)

    plt.title('Asset Changes Over Time for Each Strategy')
    plt.xlabel('Date')
    # usd TWD as y-axis and print in million unit
    plt.ylabel('Total Asset Value (TWD)')
    # max_asset = max([max(asset_values)
    #                 for asset_values in final_asset_values.values()])
    # max_step = math.ceil(max_asset / initial_investment) + 1
    # plt.yticks(np.arange(0, max_step * initial_investment, initial_investment))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{start_date}.png')
    # plt.show()

    return asset_changes


def calculate_rolling_return(assets, window_year, start_date, tradingDate):
    """
    1. Calculate the rolling return for each strategy
        - The rolling return is calculated by taking the difference between the final value and the initial value of the portfolio over a rolling window of a certain number of years (e.g., 1, 3, 5, 7, 9 years).
        - The rolling return is then divided by the initial value of the portfolio to get the percentage return.
    2. Calculate the best, worst, and average rolling return for each strategy over the specified window period.
        - Find best return and its period
        - Find worst return and its period
        - Find average return
    """
    rolling_return = {}

    with open(f"{start_date}rolling_return_{window_year}.txt", "w") as f:
        for strategy, asset in assets.items():
            stockTwii_values = asset["stock-twii"]
            stock00631L_values = asset["stock-00631L"]
            stockQQQ_values = asset["stock-QQQ"]
            stockQLD_values = asset["stock-QLD"]
            stockSPY_values = asset["stock-SPY"]
            stockSSO_values = asset["stock-SSO"]
            stockBND_values = asset["stock-BND"]
            cashTWD_values = asset["cash-TWD"]
            cashUSD_values = asset["cash-USD"]

            total_values = [twii + etf + QQQ + QLD + SPY + SSO + BND + cashTWD + cashUSD for twii, etf, QQQ, QLD, SPY, SSO, BND, cashTWD, cashUSD in zip(
                stockTwii_values, stock00631L_values, stockQQQ_values, stockQLD_values, stockSPY_values, stockSSO_values, stockBND_values, cashTWD_values, cashUSD_values)]
            total_returns = pd.Series(total_values)
            rolling_return[strategy] = total_returns.rolling(
                window=window_year*12).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])

            best_return = rolling_return[strategy].max()
            # get best return period index
            best_return_period = rolling_return[strategy].idxmax()
            # convert index to trading date start month and end month
            (best_start_date, best_end_date) = (
                tradingDate.iloc[best_return_period-window_year*12], tradingDate.iloc[best_return_period])

            worst_return = rolling_return[strategy].min()
            # get worst return period index
            worst_return_period = rolling_return[strategy].idxmin()
            # convert index to trading date start month and end month
            (worst_start_date, worst_end_date) = (
                tradingDate.iloc[worst_return_period-window_year*12], tradingDate.iloc[worst_return_period])

            average_return = rolling_return[strategy].mean()

            f.write(f"Strategy: {strategy}\n")
            f.write(f"滾動區間: {window_year}年\n")
            f.write(f"最佳報酬: {best_return:.2%}\n")
            f.write(f"最佳報酬區間: {best_start_date.date()} ~ {best_end_date.date()}\n")
            f.write(f"最差報酬: {worst_return:.2%}\n")
            f.write(f"最差報酬區間: {worst_start_date.date()} ~ {worst_end_date.date()}\n")
            f.write(f"平均報酬: {average_return:.2%}\n")
            f.write("="*40+"\n")


history = load_data("ticket.csv")
tradingDate = pd.to_datetime(history['Date'])
# Convert the index to datetime
twiiHistory = pd.DataFrame({'Date': tradingDate, 'Price': history['^TWII']})
etf00631LHistory = pd.DataFrame(
    {'Date': tradingDate, 'Price': history['00631L.TW']})
etfQQQHistory = pd.DataFrame({'Date': tradingDate, 'Price': history['QQQ']})
etfQLDHistory = pd.DataFrame({'Date': tradingDate, 'Price': history['QLD']})
etfSPYHistory = pd.DataFrame({'Date': tradingDate, 'Price': history['SPY']})
etfSSOHistory = pd.DataFrame({'Date': tradingDate, 'Price': history['SSO']})
etfBNDHistory = pd.DataFrame({'Date': tradingDate, 'Price': history['BND']})

# Calculate the percentage change of the stock
etf00631LFinal = extend_leverage_data(twiiHistory, etf00631LHistory).fillna(0)
etfQLDFinal = extend_leverage_data(etfQQQHistory, etfQLDHistory).fillna(0)
etfSSOFinal = extend_leverage_data(etfSPYHistory, etfSSOHistory).fillna(0)
twiiFinal = pd.DataFrame(
    {'Date': tradingDate, 'Percentage': twiiHistory['Price'].pct_change(fill_method=None)}).fillna(0)
etfQQQFinal = pd.DataFrame(
    {'Date': tradingDate, 'Percentage': etfQQQHistory['Price'].pct_change(fill_method=None)}).fillna(0)
etfSPYFinal = pd.DataFrame(
    {'Date': tradingDate, 'Percentage': etfSPYHistory['Price'].pct_change(fill_method=None)}).fillna(0)
etfBNDFinal = pd.DataFrame(
    {'Date': tradingDate, 'Percentage': etfBNDHistory['Price'].pct_change(fill_method=None)}).fillna(0)

for start_date in ['1997-08-01', '2000-01-01', '2004-01-01', '2014-01-01']:
    asset_changes = simulatedPortfolioChange(start_date, tradingDate[tradingDate >= start_date],
                                             1000000, 0,
                                             filter_data_by_start_date(
                                                 twiiFinal, start_date),
                                             filter_data_by_start_date(
                                                 etf00631LFinal, start_date),
                                             filter_data_by_start_date(
                                                 etfQQQFinal, start_date),
                                             filter_data_by_start_date(
                                                 etfQLDFinal, start_date),
                                             filter_data_by_start_date(
                                                 etfSPYFinal, start_date),
                                             filter_data_by_start_date(
                                                 etfSSOFinal, start_date),
                                             filter_data_by_start_date(etfBNDFinal, start_date))
    for window in [1, 3, 5, 7, 9]:
        rollong_return = calculate_rolling_return(
            asset_changes, window, start_date, tradingDate[tradingDate >= start_date])
    print(f"Simulated Portfolio Change for start date: {start_date} is done")
