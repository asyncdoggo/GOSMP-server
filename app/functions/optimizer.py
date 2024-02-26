from collections import defaultdict
import math
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier
import numpy as np
import pandas as pd
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.exceptions import OptimizationError
import datetime

def without_optimization(timed_df: pd.DataFrame):
    """
    This function calculates the portfolio variance, volatility, and annual return without optimization.
    timed_df: pd.DataFrame: The processed df with the date as the index and the stock prices as the columns.
    """

    # Show daily returns
    returns = timed_df.pct_change(fill_method=None)
    returns.fillna(0, inplace=True)
    returns.replace([np.inf, -np.inf], 0, inplace=True)

    # Annualized covariance matrix
    # cov_matrix_annual = returns.cov()*252
    cov_matrix_annual = returns.cov()*246
    cov_matrix_annual.fillna(0, inplace=True)

    # assign equal weights to each stock
    weights = np.full(timed_df.shape[1], 1 / timed_df.shape[1])

    # calculate the portfolio variance
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))

    # calculate the portfolio volatility aka standard deviation
    port_volatility = np.sqrt(port_variance)

    # calculate the annual portfolio return
    port_annual_return = np.sum(returns.mean() * weights) * 246

    # expected annual return, volatility, and variance
    percent_var = str(round(port_variance, 2) * 100) + '%'
    percent_vols = str(round(port_volatility, 2) * 100) + '%'
    percent_ret = str(round(port_annual_return, 2) * 100) + '%'

    return port_variance, port_volatility, port_annual_return, percent_var, percent_vols, percent_ret


def optimize(timed_df: pd.DataFrame, exp_ret_type: dict, cov_type: dict, weight_type: dict, invest_amount: int, sectors_map: dict, sector_lower: dict, sector_upper: dict):
    performance, refined_weights = efficient_frontier(
        timed_df, exp_ret_type, cov_type, weight_type, sectors_map, sector_lower, sector_upper)

    start_date = timed_df.index[timed_df.index.get_indexer(
        [datetime.datetime.now().date()], method='nearest')][0].date()

    # remove stocks with 0 weight
    refined_weights = {key: value for key,
                       value in refined_weights.items() if value != 0.0}

    # Normalize the percentages
    total_weight = sum(refined_weights.values())
    refined_weights_percent = {
        key: (value / total_weight) * 100 for key, value in refined_weights.items()}

    invested, remaining = _discrete_allocate(
        invest_amount, refined_weights, timed_df, start_date)

    return performance, invested, refined_weights_percent, remaining, start_date


def efficient_frontier(timed_df: pd.DataFrame, exp_ret_type: dict, cov_type: dict, weight_type: dict, sectors_map: dict, sector_lower: dict, sector_upper: dict):
    """
    timed_df: pd.DataFrame: The processed df with the date as the index and the stock prices as the columns.
    exp_ret_type: dict: The expected return type to be used in the optimization.
    cov_type: dict: The covariance type to be used in the optimization.

    exp_ret_type: dict: The expected return type to be used in the optimization.
    1. mean: The mean historical return.
    dict = {"type": "mean", "frequency": number}
    2. ema: The exponential moving average return.
    dict = {"type": "ema", "log_returns": boolean}
    3. capm: The CAPM return.
    dict = {"type": "capm"}

    cov_type: dict: The covariance type to be used in the optimization.
    1. sample_cov: The sample covariance matrix.
    dict = {"type": "sample_cov"}
    2. exp_cov: The exponentially weighted covariance matrix.
    dict = {"type": "exp_cov"}
    3. shrinkage: The shrunk covariance matrix.
    dict = {"type": "shrinkage"}


    weight_type: dict: The type of weights to be used in the optimization.
    1. max_sharpe: The maximum Sharpe ratio weights.
    dict = {"type": "max_sharpe"}
    2. min_volatility: The minimum volatility weights.
    dict = {"type": "min_volatility"}
    3. efficient_risk: The efficient risk weights.
    dict = {"type": "efficient_risk", "target_volatility": number}
    4. efficient_return: The efficient return weights.
    dict = {"type": "efficient_return", "target_return": number}
    """
    mu = None
    if exp_ret_type["type"] == "mean":
        mu = expected_returns.mean_historical_return(
            timed_df, frequency=exp_ret_type["frequency"])

    elif exp_ret_type["type"] == "ema":
        mu = expected_returns.ema_historical_return(
            timed_df, log_returns=exp_ret_type["log_returns"])

    elif exp_ret_type["type"] == "capm":
        mu = expected_returns.capm_return(timed_df)

    mu.fillna(0, inplace=True)
    mu.replace(np.inf, 0.0, inplace=True)
    S = None
    if cov_type["type"] == "sample_cov":
        S = risk_models.sample_cov(timed_df)

    elif cov_type["type"] == "exp_cov":
        S = risk_models.exp_cov(timed_df)

    elif cov_type["type"] == "shrinkage":
        S = risk_models.CovarianceShrinkage(timed_df)

    S.fillna(0, inplace=True)
    S.replace(np.inf, 0.0, inplace=True)

    # Regularize the covariance matrix
    S_f = S + 1e-6 * np.eye(S.shape[0])

    # Optimize for the maximal Sharpe ratio
    ef = EfficientFrontier(mu, S_f, solver="ECOS")

    ef.add_sector_constraints(sectors_map, sector_lower, sector_upper)

    if weight_type["type"] == "max_sharpe":
        ef.max_sharpe()
    elif weight_type["type"] == "min_volatility":
        ef.min_volatility()
    elif weight_type["type"] == "efficient_risk":
        ef.efficient_risk(
            target_volatility=weight_type["target_volatility"])
    elif weight_type["type"] == "efficient_return":
        ef.efficient_return(
            target_return=weight_type["target_return"])

    # calculate the portfolio variance
    refined_weights = ef.clean_weights()

    # get the portfolio performance and prints it
    performance = ef.portfolio_performance(verbose=True)

    return performance, refined_weights


def _discrete_allocate(invest_amount, refined_weights, timed_df, start_date):

    ndf = timed_df[refined_weights.keys()]

    # get the latest price of the stocks
    lp = ndf.loc[
        ndf.index[ndf.index.get_indexer([start_date], method='nearest')]
    ]
    lp = pd.Series(lp.values[0], index=lp.columns)
    
    da = DiscreteAllocation(refined_weights, lp,
                            total_portfolio_value=invest_amount)

    try:
        allocation, leftover = da.lp_portfolio()
    except OptimizationError as e:
        allocation, leftover = da.greedy_portfolio()

    for key, value in allocation.items():
        allocation[key] = {
            "price": lp[key],
            "units": int(value),
            "allocated": lp[key] * value
        }

    return allocation, leftover


def _allocate_nifty(invest_amount, nifty, start_date):
    lp = nifty.loc[
        nifty.index[nifty.index.get_indexer([start_date], method='nearest')]
    ].values[0][0]

    allocation, leftover = invest_amount // lp, invest_amount % lp

    return {"nifty": {"price": lp, "units": allocation, "allocated": lp * allocation}}, leftover


def BackTest(df, start_date, duration, weights ):
    """
    duration: in days
    startDate: starting date string
    weights: weights dict 
    """
    last_date = df.index[-1]
    # start = datetime.datetime.strptime(startDate, "%Y-%m-%d")
    start = start_date
    end = start + datetime.timedelta(days=duration)

    x = defaultdict(dict)
    c = 1
    while end < last_date.date() and start < last_date.date():

        end = start + datetime.timedelta(days=duration)
        temp = df.loc[start:end, :]

        for i in weights.keys():
            x[i][c] = {
                "date_start": str(temp[i].iloc[0:].index[0])[:10],
                "date_end": str(temp[i].iloc[-1:].index[0])[:10],
                "date_start_price": temp[i].iloc[0],
                "date_end_price": temp[i].iloc[-1]
            }
        for key, values in x.items():
            st = values[c]["date_start_price"]
            en = values[c]["date_end_price"]

            st_price = st * weights[key]["units"]
            en_price = en * weights[key]["units"]
            pct_cng = (en_price - st_price)/st_price * 100
            values[c]["st_alloc"] = st_price
            values[c]["en_alloc"] = en_price

            values[c]["pct_change"] = pct_cng
        start = end
        c += 1

    return x, c-1


def total_return(nifty_csv_file, start_date, backtest_duration, timed_df, invested):
    nifty = load_nifty(nifty_csv_file)

    end_invest = start_date + datetime.timedelta(days=backtest_duration)
    end_invest = timed_df.index[timed_df.index.get_indexer(
        [end_invest], method='nearest')]
    total_portfolio_return = []
    for key, value in invested.items():
        invested_start = timed_df[key][start_date.strftime(
            "%Y-%m-%d")] * invested[key]["units"]
        invested_end = timed_df[key][end_invest] * invested[key]["units"]
        total_portfolio_return.append(
            (invested_end - invested_start) / invested_start * 100)

    nifty_start = nifty["nifty"][start_date.strftime("%Y-%m-%d")]
    nifty_end = nifty["nifty"][end_invest]
    nifty_return = (nifty_end - nifty_start) / nifty_start * 100

    print(total_portfolio_return)

    return sum(total_portfolio_return) / len(total_portfolio_return), nifty_return


def get_returns(duration, start_date, invested, timed_df):
    # annual returns = (ending value / beginning value) ^ (1/years) - 1
    num_years = duration / 365

    # calculate the annual returns for each stock in invested
    annual_returns = {}
    for stock, allocation_data in invested.items():
        stock_data = timed_df[stock]
        start_allocated = stock_data.iloc[stock_data.index.get_indexer(
            [start_date], method='nearest')][0]

        end_allocated = stock_data.iloc[stock_data.index.get_indexer(
            [start_date + datetime.timedelta(days=duration)], method='nearest')][0]

        annual_returns[stock] = (
            end_allocated / start_allocated) ** (1/num_years) - 1

    # overall returns = (ending value - beginning value)/ beginning value
    overall_returns = {}
    for stock, data in invested.items():
        stock_data = timed_df[stock]
        start_allocated = stock_data.iloc[stock_data.index.get_indexer(
            [start_date], method='nearest')][0]

        end_allocated = stock_data.iloc[stock_data.index.get_indexer(
            [start_date + datetime.timedelta(days=duration)], method='nearest')][0]

        overall_returns[stock] = (
            end_allocated - start_allocated) / start_allocated

    return annual_returns, overall_returns


def _percent_change(window, totalWindows):
    pctChange = []
    endDate = []
    for part in range(1, totalWindows+1):
        startPrice = endPrice = 0
        end = None
        for key, value in window.items():
            cycle = value.get(part)
            startPrice += cycle['st_alloc']
            endPrice += cycle['en_alloc']
            end = cycle['date_end']
        endDate.append(end)
        pctChange.append(((endPrice - startPrice)/startPrice * 100))
    return pctChange, endDate


def backtest_with_nifty(timed_df, invest_amount, invested, weights, duration):
    newTimeDf = timed_df[[i for i in invested.keys()]]

    nifty = load_nifty("app/data/nifty.csv")

    start_date = datetime.datetime.now().date() - datetime.timedelta(days=3 * 365)

    start_date = timed_df.index[timed_df.index.get_indexer(
        [start_date], method='nearest')][0].date()

    invested_nifty, remainder_nifty = _allocate_nifty(
        invest_amount, nifty, start_date)

    window, total_windows = BackTest(
        newTimeDf, start_date, duration, invested)

    win, total_ = BackTest(nifty, start_date, duration, invested_nifty)

    portfolioPercentChange, endDates = _percent_change(window, total_windows)
    niftyPercentChange, niftyendDates = _percent_change(win, total_)

    if len(portfolioPercentChange) > len(niftyPercentChange):
        portfolioPercentChange = portfolioPercentChange[:len(
            niftyPercentChange)]
        endDates = endDates[:len(niftyPercentChange)]

    results = pd.DataFrame({
        'Date': endDates,
        'PctChange': portfolioPercentChange,
        'niftyPctChange': niftyPercentChange
    })

    return results, invested_nifty


def load_nifty(nifty_csv_file: str):
    nifty = pd.read_csv(nifty_csv_file)

    nifty['Date'] = pd.to_datetime(nifty['Date'])

    nifty.set_index('Date', inplace=True)

    # Drop columns where every entry is 0.0
    nifty = nifty.loc[:, (nifty != 0).any(axis=0)]

    # # # Use the column selection to drop columns where less than the threshold number of values are non-zero
    threshold = 0.70 * len(nifty)
    nifty = nifty.loc[:, (nifty != 0).sum() >= threshold]
    nifty = nifty.iloc[::-1]

    # rename Close to Nifty
    nifty = nifty.rename(columns={'Close': 'nifty'})

    # reverse the index
    nifty = nifty.iloc[::-1]

    return nifty


def calculate_annual_returns(invest_amount, duration, start_date, invested, timed_df):
    # divide start date and end date into 365 days
    # for each year, calculate the return
    # return the total return

    # invested = {'INDRAMEDCO': {'price': 36.45000076293945,
    #   'units': 2743,
    #   'allocated': 99982.35209274292},
    #  'PADAMCO': {'price': 10.729999542236328,
    #   'units': 1,
    #   'allocated': 10.729999542236328},
    #  'VINTRON': {'price': 0.550000011920929,
    #   'units': 12,
    #   'allocated': 6.6000001430511475}}

    total_return = {

    }
    start_date = timed_df.index[timed_df.index.get_indexer(
        [start_date], method='nearest')][0].date()

    for i in range(0, duration, 365):
        end_date = start_date + datetime.timedelta(days=365)
        end_date = min(end_date, datetime.datetime.now().date())

        total_return[f"year_{i}"] = {
            "start_date": start_date,
            "end_date": end_date,
            "return": {},
            "total": 0
        }
        year_return = {}
        for stock, allocation_data in invested.items():
            stock_data = timed_df[stock]
            start_price = stock_data.iloc[stock_data.index.get_indexer(
                [start_date], method='nearest')][0]

            end_price = stock_data.iloc[stock_data.index.get_indexer(
                [end_date], method='nearest')][0]
            pct_change = (end_price - start_price) / start_price
            year_return[stock] = pct_change
            total_return[f"year_{i}"]["total"] += pct_change
        total_return[f"year_{i}"]["return"] = year_return
        start_date = end_date

    return total_return

# actual = calculate_annual_returns(invest_amount, duration, start_date, invested, timed_df)
