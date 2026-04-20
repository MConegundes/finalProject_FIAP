from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from enum import Enum
from datetime import date
import numpy as np
from typing import Optional

from src.inferencia import load_artifacts
from src.train import train_model
from api.train_status import TRAIN_STATUS 
from src.data_loader import load_data

from datetime import timedelta
from src.utils.prediction_saver import save_predictions_csv

app = FastAPI(title="TESLA, BYD & TOYOTA LSTM Predictive API")



# ===============================
# ENUM DE SÍMBOLOS
# ===============================
class StockSymbol(str, Enum):
    TSLA = "TSLA"
    BYD = "BYD"
    TOYOTA = "TOYOTA"

# ===============================
# MAPEAMENTO PARA Yahoo Finance
# ===============================
SYMBOLS_MAP = {
    StockSymbol.TSLA: "TSLA",
    StockSymbol.BYD: "BYDDY",
    StockSymbol.TOYOTA: "TM"
}

# ===============================
# SCHEMAS
# ===============================
class PredictRequest(BaseModel):
    symbol: StockSymbol
    days_ahead: int = Field(default=5, ge=1, le=60)
    start_date: date = Field(default=date(2015, 1, 1))
    end_date: date = Field(default=date(2025, 12, 31))

    @field_validator("end_date")
    @classmethod
    def check_dates(cls, v: date, info: ValidationInfo):
        start_date = info.data.get("start_date")

        if start_date and v < start_date:
            raise ValueError("A data de término deve ser maior ou igual à data de início.")

        return v

class TrainRequest(BaseModel):
    symbol: StockSymbol
    start_date: Optional[date] = Field(default=date(2015, 1, 1))
    end_date: Optional[date] = Field(default=date(2025, 12, 31))
    epochs: Optional[int] = Field(default=20, ge=1, le=100)


# ===============================
# ENDPOINT /predict
# ===============================
@app.post("/predict")
def predict(req: PredictRequest):
    symbol_enum = req.symbol
    symbol = symbol_enum.value
    ticker = SYMBOLS_MAP[symbol_enum]

    # Carregar modelo e scaler
    try:
        model, scaler = load_artifacts(symbol)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Carregar histórico de preços
    df = load_data(
        ticker,
        start_date=req.start_date.isoformat(),
        end_date=req.end_date.isoformat()
    )

    if df.empty:
        raise HTTPException(status_code=404, detail="Nenhum dado disponível para o período informado")

    # Pegar apenas coluna de fechamento
    values = df[['Close']].values
    scaled = scaler.transform(values)

    # Preparar janela de input
    window_size = model.input_shape[1]
    last_window = scaled[-window_size:].reshape(1, window_size, 1)

    # Previsão iterativa
    preds_scaled = []
    for _ in range(req.days_ahead):
        pred = model.predict(last_window, verbose=0)
        preds_scaled.append(pred[0, 0])
        last_window = np.append(
            last_window[:, 1:, :],
            [[[pred[0, 0]]]],
            axis=1
        )

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled)

    # Datas futuras
    last_date = df["Date"].iloc[-1]

    future_dates = [
        (last_date + timedelta(days=i)).date()
        for i in range(1, req.days_ahead + 1)
    ]

    # Salvar previsões em CSV
    file_path = save_predictions_csv(
        symbol=symbol,
        dates=future_dates,
        real_values=None,
        predicted_values=preds.flatten().tolist(),
        model_version="v1"
    )

    return {
    "symbol": symbol,
    "ticker": ticker,
    "days_ahead": req.days_ahead,
    "start_date": req.start_date.isoformat(),
    "end_date": req.end_date.isoformat(),
    "predictions": preds.flatten().tolist(),
    "csv_saved_at": str(file_path)
}


# ===============================
# ENDPOINT /train (background task)
# ===============================
@app.post("/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    symbol_enum = req.symbol
    symbol = symbol_enum.value
    ticker = SYMBOLS_MAP[symbol_enum]

    # Inicializa status do treino
    TRAIN_STATUS[symbol] = {"progress": 0, "status": "Iniciado"}

    # Função em background
    def run_training(symbol, ticker, start_date, end_date, epochs):
        try:
            result = train_model(
                symbol=symbol,
                ticker=ticker,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                epochs=epochs
            )
            TRAIN_STATUS[symbol]["progress"] = 100
            TRAIN_STATUS[symbol]["status"] = "Concluído"
        except Exception as e:
            TRAIN_STATUS[symbol]["status"] = f"Erro: {str(e)}"
            TRAIN_STATUS[symbol]["progress"] = 0

    # Executa treino em background
    background_tasks.add_task(run_training, symbol, ticker, req.start_date, req.end_date, req.epochs)

    return {"message": f"Treinamento de {symbol} iniciado em background"}


# ===============================
# STATUS DO TREINO
# ===============================
@app.get("/train/status/{symbol}")
def train_status(symbol: StockSymbol):
    symbol_str = symbol.value
    if symbol_str not in TRAIN_STATUS:
        raise HTTPException(status_code=404, detail="Treino não encontrado")
    return TRAIN_STATUS[symbol_str]
