import yfinance as yf
import pandas as pd
from datetime import datetime

SYMBOLS = {
    "TSLA": "TSLA",
    "BYD": "BYDDY",
    "TOYOTA": "TM"
}



def load_data(ticker: str, start_date: str = "2015-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    """
    Baixa dados históricos de preços de ações via Yahoo Finance.
    
    Args:
        ticker (str): Código da ação (ex: 'TSLA').
        start_date (str): Data inicial no formato 'YYYY-MM-DD'.
        end_date (str): Data final no formato 'YYYY-MM-DD'.
    
    Returns:
        pd.DataFrame: DataFrame com colunas ['Open', 'High', 'Low', 'Close', 'Volume'].
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return pd.DataFrame()  # Retorna DataFrame vazio caso não haja dados
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.reset_index(inplace=True)
    return df



def load_all(start_date: str, end_date: str) -> dict:
    """
    Carrega dados para Tesla, BYD e Toyota.

    Returns:
        dict: {empresa: dataframe}
    """
    data = {}
    for company, ticker in SYMBOLS.items():
        data[company] = load_data(ticker, start_date, end_date)
    return data


if __name__ == "__main__":
    data = load_all("2015-01-01", datetime.today().strftime("%Y-%m-%d"))
    for k, v in data.items():
        print(k, v.tail())