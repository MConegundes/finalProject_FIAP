import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
import os

MODEL_DIR = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_artifacts(symbol: str):
    """Carrega modelo e scaler salvos para o símbolo"""
    model_path = os.path.join(MODEL_DIR, f"lstm_{symbol}.keras")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{symbol}.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Modelo ou scaler não encontrados para {symbol}")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler