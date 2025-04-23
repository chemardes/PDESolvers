import numpy as np
import yfinance as yf

class HistoricalStockData:
    def __init__(self, ticker):
        self.__stock_data = None
        self.__ticker = ticker

    def fetch_stock_data(self, start_date, end_date):
        self.__stock_data = yf.download(self.__ticker, start=start_date, end=end_date, interval='1d')
        return self.__stock_data

    def estimate_metrics(self):

        if self.__stock_data is None:
            raise ValueError(f'No data available - data must be fetched first.')

        self.__stock_data["Log Returns"] = np.log(self.__stock_data["Close"] / self.__stock_data["Close"].shift(1))

        # takes into account the gaps between trading days
        self.__stock_data["Time Diff"] = self.__stock_data.index.to_series().diff().dt.days
        self.__stock_data["Z"] = self.__stock_data["Log Returns"] / self.__stock_data["Time Diff"]

        sigma = self.__stock_data["Log Returns"].std() * np.sqrt(252)
        mu = self.__stock_data["Log Returns"].mean() * 252

        return sigma, mu

    def get_latest_stock_price(self):
        if self.__stock_data is None:
            raise ValueError("No data available. Call fetch_data first.")

        return self.__stock_data["Close"].iloc[-1].item()

    def get_initial_stock_price(self):
        if self.__stock_data is None:
            raise ValueError("No data available. Call fetch_data first.")

        return self.__stock_data["Close"].iloc[0].item()

    def get_closing_prices(self):
        if self.__stock_data is None:
            raise ValueError("No data available. Call fetch_data first.")

        return self.__stock_data["Close"].values