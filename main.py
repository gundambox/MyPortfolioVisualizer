# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

DOLLOR_TO_TWD = 32.07
ticket_list = ["^TWII", "00631L.TW", "SPY", "SSO"]

def download_data(path):
    # Download the data from Yahoo Finance
    history = yf.download(ticket_list, period='max', ignore_tz=True)
    history = history['Close']
    history.to_csv(path)
    return history

def load_data(path):
    if os.path.exists(path):
        history = pd.read_csv(path)
        # Convert column name 'Ticker' to 'Date' and set it as index
        last_trading_day = pd.to_datetime(history['Date']).max().to_pydatetime().date()
        # if last trading day is not today, download the latest data
        if last_trading_day < (datetime.datetime.today() + datetime.timedelta(days=-1)).date():
            print("Download the latest data")
            history = download_data(path)
    else:
        history = download_data(path)

    return history

def extend_leverage_data(origin_data, leverage_data):
    origin_pct_change_data = origin_data['Price'].pct_change(fill_method=None)
    leverage_pct_change_data = leverage_data['Price'].pct_change(fill_method=None)
    np.random.seed(0)

    # Find the first valid index of the origin_pct_change_data
    origin_pct_change_data_idx = pd.Series(origin_pct_change_data).first_valid_index()
    # Find the first valid index of the leverage_pct_change_data
    leverage_pct_change_data_idx = pd.Series(leverage_pct_change_data).first_valid_index()

    # pick the range from origin_pct_change_data_idx to leverage_pct_change_data_idx and multiply 2 and reassign back to leverage_pct_change_data
    leverage_final = pd.concat([origin_pct_change_data[:origin_pct_change_data_idx],
                                origin_pct_change_data[origin_pct_change_data_idx:leverage_pct_change_data_idx] * 2,
                                leverage_pct_change_data[leverage_pct_change_data_idx:]])
    leverage_final = pd.DataFrame({'Date': origin_data['Date'], 'Percentage': leverage_final})
    return leverage_final

def prompt_user_input_date():
    # start_date = input("Enter the start date (YYYY-MM-DD): ")
    # # if start date is not provided, return default start date
    # if not start_date:
    # return "1997-07-03"
    return "2000-01-01"
    # return "2004-01-01"
    # return "2014-01-01"
    # else:
    #     return start_date

def filter_data_by_start_date(data, start_date):
    return data[data['Date']>=start_date]

def rebalance_portfolio(asset_changes, strategy, allocation):
    asset_cashTWD = asset_changes[strategy]["cash-TWD"][-1]
    asset_cashUSD = asset_changes[strategy]["cash-USD"][-1]
    asset_stockWwii = asset_changes[strategy]["stock-twii"][-1]
    asset_stock00631L = asset_changes[strategy]["stock-00631L"][-1]
    asset_stockSpy = asset_changes[strategy]["stock-SPY"][-1]
    asset_stockSso = asset_changes[strategy]["stock-SSO"][-1]
    total_value = asset_cashTWD + asset_cashUSD + asset_stockWwii + asset_stock00631L + asset_stockSpy + asset_stockSso

    cashTWD_allocation = total_value * allocation["cash-TWD"]
    cashUSD_allocation = total_value * allocation["cash-USD"]
    twii_allocation = total_value * allocation["stock-twii"]
    etf_allocation = total_value * allocation["stock-00631L"]
    spy_allocation = total_value * allocation["stock-SPY"]
    sso_allocation = total_value * allocation["stock-SSO"]

    # Update last record asset changes, because we are using currency, we round to integer
    asset_changes[strategy]["cash-TWD"][-1] = int(cashTWD_allocation)
    asset_changes[strategy]["cash-USD"][-1] = int(cashUSD_allocation)
    asset_changes[strategy]["stock-twii"][-1] = int(twii_allocation)
    asset_changes[strategy]["stock-00631L"][-1] = int(etf_allocation)
    asset_changes[strategy]["stock-SPY"][-1] = int(spy_allocation)
    asset_changes[strategy]["stock-SSO"][-1] = int(sso_allocation)

def simulatedPortfolioChange(start_date, tradingDate, initial_investment, twiiFinal, etf00631LFinal, etfSpyFinal, etfSsoFinal):
    # Filter the data by start date
    tradingDate = tradingDate[tradingDate >= start_date]

    twiiFinal = filter_data_by_start_date(twiiFinal, start_date)
    etf00631LFinal = filter_data_by_start_date(etf00631LFinal, start_date)
    etfSpyFinal = filter_data_by_start_date(etfSpyFinal, start_date)
    etfSsoFinal = filter_data_by_start_date(etfSsoFinal, start_date)

    etf00631LFinal.to_csv("00631L.TW.csv")
    etfSsoFinal.to_csv("SSO.csv")
    twiiFinal.to_csv("^TWII.csv")
    etfSpyFinal.to_csv("SPY.csv")

    initial_investment = 1200000
    strategies = {
        # All in twii
        "Strategy 1": {"cash-TWD": 0.0, "cash-USD": 0.0, "stock-twii": 1.0, "stock-00631L": 0.0, "stock-SPY": 0.0, "stock-SSO": 0.0},
        # All in 00631L
        "Strategy 2": {"cash-TWD": 0.0, "cash-USD": 0.0, "stock-twii": 0.0, "stock-00631L": 1.0, "stock-SPY": 0.0, "stock-SSO": 0.0},
        # All in SPY
        "Strategy 3": {"cash-TWD": 0.0, "cash-USD": 0.0, "stock-twii": 0.0, "stock-00631L": 0.0, "stock-SPY": 1.0, "stock-SSO": 0.0},
        # All in SSO
        "Strategy 4": {"cash-TWD": 0.0, "cash-USD": 0.0, "stock-twii": 0.0, "stock-00631L": 0.0, "stock-SPY": 0.0, "stock-SSO": 1.0},
        # 40% cash-TWD, 10% cash-USD, 40% 00631L, 10% SSO
        "Strategy 5": {"cash-TWD": 0.4, "cash-USD": 0.1, "stock-twii": 0.0, "stock-00631L": 0.4, "stock-SPY": 0.0, "stock-SSO": 0.1},
        # 35% cash-TWD, 15% cash-USD, 35% 00631L, 15% SSO
        "Strategy 6": {"cash-TWD": 0.35, "cash-USD": 0.15, "stock-twii": 0.0, "stock-00631L": 0.35, "stock-SPY": 0.0, "stock-SSO": 0.15},
        # 30% cash-TWD, 20% cash-USD, 30% 00631L, 20% SSO
        "Strategy 7": {"cash-TWD": 0.3, "cash-USD": 0.2, "stock-twii": 0.0, "stock-00631L": 0.3, "stock-SPY": 0.0, "stock-SSO": 0.2},
        # 25% cash-TWD, 25% cash-USD, 25% 00631L, 25% SSO
        "Strategy 8": {"cash-TWD": 0.25, "cash-USD": 0.25, "stock-twii": 0.0, "stock-00631L": 0.25, "stock-SPY": 0.0, "stock-SSO": 0.25}
    }

    # Initialize the asset changes for each strategy
    asset_changes = { strategy: {asset: [int(initial_investment*ratio)] for asset, ratio in allocation.items()} for strategy, allocation in strategies.items() }

    # Allocate assets based on each strategy first trading day of each year
    last_year = datetime.datetime.strptime(start_date, "%Y-%m-%d").year - 1

    for date in tradingDate:
        # Check if the year has changed
        if pd.to_datetime(date).year != last_year:
            # Rebalance the portfolio
            print(f"First trading day of the year {pd.to_datetime(date).year}, rebalance the portfolio")
            for strategy, allocation in strategies.items():
                # reallocate assets based on the strategy
                asset_cashTWD = asset_changes[strategy]["cash-TWD"][-1]
                asset_cashUSD = asset_changes[strategy]["cash-USD"][-1]
                asset_stockWwii = asset_changes[strategy]["stock-twii"][-1]
                asset_stock00631L = asset_changes[strategy]["stock-00631L"][-1]
                asset_stockSpy = asset_changes[strategy]["stock-SPY"][-1]
                asset_stockSso = asset_changes[strategy]["stock-SSO"][-1]
                total_value = asset_cashTWD + asset_cashUSD + asset_stockWwii + asset_stock00631L + asset_stockSpy + asset_stockSso

                cashTWD_allocation = total_value * allocation["cash-TWD"]
                cashUSD_allocation = total_value * allocation["cash-USD"]
                twii_allocation = total_value * allocation["stock-twii"]
                etf_allocation = total_value * allocation["stock-00631L"]
                spy_allocation = total_value * allocation["stock-SPY"]
                sso_allocation = total_value * allocation["stock-SSO"]

                # Update last record asset changes, because we are using currency, we round to integer
                asset_changes[strategy]["cash-TWD"][-1] = int(cashTWD_allocation)
                asset_changes[strategy]["cash-USD"][-1] = int(cashUSD_allocation)
                asset_changes[strategy]["stock-twii"][-1] = int(twii_allocation)
                asset_changes[strategy]["stock-00631L"][-1] = int(etf_allocation)
                asset_changes[strategy]["stock-SPY"][-1] = int(spy_allocation)
                asset_changes[strategy]["stock-SSO"][-1] = int(sso_allocation)
            last_year = pd.to_datetime(date).year

        for strategy, allocation in strategies.items():
            # Get the percentage change of the stock according to the date, if this date does not have trading data, assume the stock price does not change
            twii_change = twiiFinal.loc[twiiFinal['Date'] == date, 'Percentage']
            asset_changes[strategy]["stock-twii"].append(int(asset_changes[strategy]["stock-twii"][-1] * (1 + twii_change)))

            etf00631L_change = etf00631LFinal.loc[etf00631LFinal['Date'] == date, 'Percentage']
            asset_changes[strategy]["stock-00631L"].append(int(asset_changes[strategy]["stock-00631L"][-1] * (1 + etf00631L_change)))

            etfSpy_change = etfSpyFinal.loc[etfSpyFinal['Date'] == date, 'Percentage']
            asset_changes[strategy]["stock-SPY"].append(int(asset_changes[strategy]["stock-SPY"][-1] * (1 + etfSpy_change)))

            etfSso_change = etfSsoFinal.loc[etfSsoFinal['Date'] == date, 'Percentage']
            asset_changes[strategy]["stock-SSO"].append(int(asset_changes[strategy]["stock-SSO"][-1] * (1 + etfSso_change)))

            asset_changes[strategy]["cash-TWD"].append(int(asset_changes[strategy]["cash-TWD"][-1]))
            asset_changes[strategy]["cash-USD"].append(int(asset_changes[strategy]["cash-USD"][-1]))

    # Save the asset changes to csv
    min_length = len(tradingDate)
    # for strategy, allocation in asset_changes.items():
    #     min_length = min(len(tradingDate), len(allocation['cash-TWD']), len(allocation['cash-USD']), len(allocation['stock-twii']), len(allocation['stock-00631L']), len(allocation['stock-SPY']), len(allocation['stock-SSO']))
    #     tradingDate = tradingDate[:min_length]
    #     df = pd.DataFrame({'Date': tradingDate,
    #                         'cash-TWD': allocation["cash-TWD"][0:min_length],
    #                         'cash-USD': allocation["cash-USD"][0:min_length],
    #                         'stock-twii': allocation["stock-twii"][0:min_length],
    #                         'stock-00631L': allocation["stock-00631L"][0:min_length],
    #                         'stock-SPY': allocation["stock-SPY"][0:min_length],
    #                         'stock-SSO': allocation["stock-SSO"][0:min_length]})
    #     df.to_csv(f"{strategy}.csv")

    # For each strategy, calculate the annualized return, standard deviation, Sharpe ratio, maximum drawdown, maximum consecutive gain days, and maximum consecutive loss days
    # Print the metrics for each strategy
    # Save report to file
    with open(f"{start_date}_report.txt", "w") as f:
        for strategy, allocation in strategies.items():
            print(f"Strategy: {strategy}")
            stockTwii_values = asset_changes[strategy]["stock-twii"]
            stock00631L_values = asset_changes[strategy]["stock-00631L"]
            stockSpy_values = asset_changes[strategy]["stock-SPY"]
            stockSso_values = asset_changes[strategy]["stock-SSO"]
            cashTWS_values = asset_changes[strategy]["cash-TWD"]
            cashUSD_values = asset_changes[strategy]["cash-USD"]

            total_values = [twii + etf + spy + sso + cashTWD + cashUSD for twii, etf, spy, sso, cashTWD, cashUSD in zip(stockTwii_values, stock00631L_values, stockSpy_values, stockSso_values, cashTWS_values, cashUSD_values)]
            total_returns = pd.Series(total_values)
            cumulative_return = (total_returns.iloc[-1] - total_returns.iloc[0]) / total_returns.iloc[0]
            annualized_return = (total_returns.iloc[-1] / total_returns.iloc[0]) ** (252 / len(total_returns)) - 1
            annualized_std = total_returns.pct_change().std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_std
            max_drawdown = (total_returns / total_returns.cummax() - 1).min()

            print(f"總報酬: {cumulative_return:.2%}")
            print(f"年化報酬: {annualized_return:.2%}")
            print(f"年化標準差: {annualized_std:.2%}")
            print(f"最大跌幅: {max_drawdown:.2%}")
            print("="*40)

            f.write(f"Strategy: {strategy}\n")
            f.write(f"總報酬: {cumulative_return:.2%}\n")
            f.write(f"年化報酬: {annualized_return:.2%}\n")
            f.write(f"年化標準差: {annualized_std:.2%}\n")
            f.write(f"最大跌幅: {max_drawdown:.2%}\n")
            f.write("="*40+"\n")

    # Step 7: Plot Asset Changes
    # Create a visualization of asset changes over time for each strategy
    # Save the plot to a file

    plt.figure(figsize=(14, 7))

    for strategy, allocation in strategies.items():
        stockTwii_values = asset_changes[strategy]["stock-twii"][:min_length]
        stock00631L_values = asset_changes[strategy]["stock-00631L"][:min_length]
        stockSpy_values = asset_changes[strategy]["stock-SPY"][:min_length]
        stockSso_values = asset_changes[strategy]["stock-SSO"][:min_length]
        cashTWS_values = asset_changes[strategy]["cash-TWD"][:min_length]
        cashUSD_values = asset_changes[strategy]["cash-USD"][:min_length]

        total_values = [twii + etf + spy + sso + cashTWD + cashUSD for twii, etf, spy, sso, cashTWD, cashUSD in zip(stockTwii_values, stock00631L_values, stockSpy_values, stockSso_values, cashTWS_values, cashUSD_values)]
        plt.plot(tradingDate, total_values, label=strategy)

    plt.title('Asset Changes Over Time for Each Strategy')
    plt.xlabel('Date')
    plt.ylabel('Total Asset Value (TWD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{start_date}.png')
    # plt.show()

history = load_data("ticket.csv")
tradingDate = pd.to_datetime(history['Date'])
# Convert the index to datetime
twiiHistory =  pd.DataFrame({'Date':tradingDate, 'Price': history['^TWII']})
etf00631LHistory = pd.DataFrame({'Date':tradingDate, 'Price': history['00631L.TW']})
etfSpyHistory = pd.DataFrame({'Date':tradingDate, 'Price': history['SPY']})
etfSsoHistory = pd.DataFrame({'Date':tradingDate, 'Price': history['SSO']})

# Calculate the percentage change of the stock
etf00631LFinal = extend_leverage_data(twiiHistory, etf00631LHistory).fillna(0)
etfSsoFinal = extend_leverage_data(etfSpyHistory, etfSsoHistory).fillna(0)
twiiFinal = pd.DataFrame({'Date':tradingDate, 'Percentage': twiiHistory['Price'].pct_change(fill_method=None)}).fillna(0)
etfSpyFinal = pd.DataFrame({'Date':tradingDate, 'Percentage': etfSpyHistory['Price'].pct_change(fill_method=None)}).fillna(0)

for start_date in ['1997-07-03', '2000-01-01', '2004-01-01', '2014-01-01']:
    simulatedPortfolioChange(start_date, tradingDate, 1200000, twiiFinal, etf00631LFinal, etfSpyFinal, etfSsoFinal)
    print(f"Simulated Portfolio Change for start date: {start_date} is done")