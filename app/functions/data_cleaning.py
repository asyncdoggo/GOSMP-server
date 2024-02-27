import os
import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")


def setup():
    # check if Equity.csv exists
    if not os.path.exists(os.path.join("app", "data", "ind_nifty500list.csv")):
        print("script code csv not found. Please download the file and place it in the data directory with name 'ind_nifty500list.csv'")
        exit(1)

    # check if close_prices.csv exists
    _download_close_prices()


def load_and_clean(timed_df, risk_cat):
    timed_df.index = pd.to_datetime(timed_df.index)
    # Drop columns where every entry is 0.0
    timed_df = timed_df.loc[:, (timed_df != 0).any(axis=0)]

    # # # Use the column selection to drop columns where less than the threshold number of values are non-zero
    threshold = 0.70 * len(timed_df)
    timed_df = timed_df.loc[:, (timed_df != 0).sum() >= threshold]

    timed_df = timed_df.replace(0, np.nan).ffill().bfill()
    # Iterate through each column
    for col in timed_df.columns:
        # Calculate the mean of the last 10 non-zero values using rolling and mean
        rolling_mean = timed_df[col].replace(
            0, np.nan).rolling(window=10, min_periods=1).mean()

        # Fill zero values with the calculated rolling mean
        timed_df[col] = timed_df.apply(
            lambda row: row[col] if row[col] != 0 else rolling_mean[row.name], axis=1)

    timed_df = _risk_categorize(timed_df, risk_cat)

    return timed_df


def _risk_categorize(timed_df, risk_cat):

    # classify stocks into risk categories based on volatility
    volatility = timed_df.pct_change().std()
    risk_categories = pd.qcut(volatility, 4, labels=[
                              'Low risk', 'Moderate risk', 'High risk', 'Very high risk'])

    risk_df = pd.DataFrame(risk_categories, columns=["Risk"])

    # risk_df["Volatility"] = volatility

    # return stocks with the specified risk category
    risk_stocks = risk_df[risk_df['Risk'] == risk_cat]
    timed_df = timed_df[risk_stocks.index]
    return timed_df


def _download_close_prices():
    inactive_count = 0
    script_code_path = os.path.join("app", "data", "ind_nifty500list.csv")
    strip_code_data = pd.read_csv(script_code_path)
    close_prices_df = None
    start_date = None
    end_date = None
    save = True
    if os.path.exists(os.path.join("app", "data", "nifty500_data.csv")):
        close_prices_df = pd.read_csv(
            os.path.join("app", "data", "nifty500_data.csv"), index_col=0)
        close_prices_df.index = pd.to_datetime(close_prices_df.index)
        # close_prices_df.Date = pd.to_datetime(close_prices_df.Date)
        # close_prices_df.set_index("Date", inplace=True)
        end_date = datetime.datetime.now().date() - datetime.timedelta(days=1)
        start_date = close_prices_df.index[-1].date() + \
            datetime.timedelta(days=1)

        # make sure start and end dates are not weekends
        # while start_date.weekday() in [5, 6]:
        #     start_date -= datetime.timedelta(days=1)
        # while end_date.weekday() in [5, 6]:
        #     end_date -= datetime.timedelta(days=1)
    else:
        end_date = datetime.datetime.now().date() - datetime.timedelta(days=1)
        start_date = end_date.replace(year=end_date.year - 10)

        # make sure start and end dates are not weekends
        # while start_date.weekday() in [5, 6]:
        #     start_date -= datetime.timedelta(days=1)
        # while end_date.weekday() in [5, 6]:
        #     end_date -= datetime.timedelta(days=1)

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        date_range = date_range[date_range.weekday < 5]

        close_prices_df = pd.DataFrame(index=date_range)
    if start_date == end_date:
        print("No new data to fetch")
        return
    count = len(strip_code_data["Symbol"])
    close_prices_df.index = pd.to_datetime(close_prices_df.index).date

    iterator = tqdm(strip_code_data["Symbol"], total=count, unit="stock")
    save = True
    for symbol in iterator:
        iterator.set_description("Processing " + symbol +
                                 " " + str(count) + " remaining")
        try:
            d = yf.download(symbol + ".NS", start=start_date,
                            end=end_date, progress=False)  # get quote here

            # check if dataframe has no rows
            if d.empty:  # if d is empty, then the stock is inactive or date range is invalid
                save = False
                break
            elif d is not None:
                # update the close prices dataframe
                d = d[["Close"]]
                d.columns = [symbol]
                d.index = pd.to_datetime(d.index).date
                close_prices_df = close_prices_df.combine_first(d)
            else:
                inactive_count += 1
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            inactive_count += 1
        count -= 1

    if save:
        close_prices_df.rename_axis("Date", inplace=True)
        sorted_close_prices_df = close_prices_df.sort_index()
        # remove duplicate index
        sorted_close_prices_df = sorted_close_prices_df[~sorted_close_prices_df.index.duplicated(
            keep='first')]

        sorted_close_prices_df.to_csv(os.path.join(
            "app", "data", "nifty500_data.csv"), index=True)


if __name__ == "__main__":
    # _download_close_prices()
    # print("done")
    # print(_risk_categorize())
    df = pd.read_csv("app/data/nifty500_data.csv")
    print(load_and_clean(df, "Low risk"))