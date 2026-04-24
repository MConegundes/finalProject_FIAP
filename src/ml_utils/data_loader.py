import yfinance as yf
import pandas as pd
from datetime import datetime
import os

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
    print('chama save data')
    save_data_db(df, ticker, 'raw')

    return df


def save_data_db(df: pd.DataFrame, ticker: str, origin: str):
    """
    Salva o DataFrame na pasta data/raw como CSV.

    Args:
        df (pd.DataFrame): DataFrame com os dados
        ticker (str): Nome do ticker (ex: 'TSLA')
    """
    # Garante que a pasta existe
    db_path = os.path.join('data', f"{origin}")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f'DB não encontrado')
    else:
        # Nome do arquivo
        file_path = os.path.join(db_path, f"{ticker}_{origin}.csv")
        # Salva o CSV
        df.to_csv(file_path, index=False)



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