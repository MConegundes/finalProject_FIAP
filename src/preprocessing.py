import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

SCALER_DIR = "data/models"
os.makedirs(SCALER_DIR, exist_ok=True)


def create_sequences(df, window_size: int = 60):
    """
    Cria sequências para treino de LSTM a partir do DataFrame de preços.
    
    Args:
        df (pd.DataFrame): DataFrame com coluna 'Close'.
        window_size (int): Número de dias para janela da LSTM.
    
    Returns:
        X (np.ndarray): Sequências de entrada (n_samples, window_size, 1)
        y (np.ndarray): Valores alvo (n_samples, 1)
        scaler (MinMaxScaler): Objeto scaler para desnormalização
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame deve conter a coluna 'Close'")

    values = df[['Close']].values.astype('float32')

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i])
    
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler



def save_scaler(scaler, symbol: str):
    """Salva o scaler para o símbolo"""
    path = os.path.join(SCALER_DIR, f"{symbol}_scaler.save")
    joblib.dump(scaler, path)

def load_scaler(symbol: str):
    """Carrega o scaler salvo para o símbolo"""
    path = os.path.join(SCALER_DIR, f"{symbol}_scaler.save")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler para {symbol} não encontrado. Treine primeiro.")
    return joblib.load(path)

def inverse_scale(data, symbol: str):
    """Reverte o scale dos dados"""
    scaler = load_scaler(symbol)
    data = np.array(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return scaler.inverse_transform(data)
