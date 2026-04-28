import logging
from typing import Any

import numpy as np
import yfinance as yf
from langchain.tools import Tool

from src.ml_utils.data_loader import load_data
from src.ml_utils.inferencia import load_artifacts
from src.utils.prediction_saver import save_predictions_csv
from datetime import date, timedelta
from fastapi import HTTPException

from src.agente.rag import retrieve_context


# ===============================
# ENUM DE SÍMBOLOS
# ===============================
class StockSymbol(str, Enum):
	TSLA = 'TSLA'
	BYD = 'BYD'
	TOYOTA = 'TOYOTA'

# ===============================
# MAPEAMENTO PARA Yahoo Finance
# ===============================
SYMBOLS_MAP = {StockSymbol.TSLA: 'TSLA', StockSymbol.BYD: 'BYDDY', StockSymbol.TOYOTA: 'TM'}


logger = logging.getLogger(__name__)

def _stock_lookup(ticker: StockSymbol) -> str:
    """Consulta dados recentes de uma ação via Yahoo Finance."""
    ticker = SYMBOLS_MAP[ticker]

    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return f"Sem dados encontrados."

        latest = hist.iloc[-1]
        info = yf.Ticker(ticker).info
        return (
            f"Ticker: {ticker}\n"
            f"Preço atual: ${latest['Close']}\n"
            f"Variação dia: {((latest['Close'] - latest['Open']) / latest['Open'] * 100):.2f}%\n"
            f"Volume: {int(latest['Volume'])}"
        )
    except Exception as e:
        return f"Erro ao consultar: {e}"


def _technical_analysis(ticker: StockSymbol) -> str:
    """Calcula indicadores técnicos para uma ação."""
    ticker = SYMBOLS_MAP[ticker]

    try:
        df = yf.download(ticker, period="3mo", progress=False)
        if df.empty:
            return f"Sem dados para análise."

        close = df["Close"]
        sma_20 = float(close.rolling(20).mean().iloc[-1])
        sma_50 = float(close.rolling(50).mean().iloc[-1])
        current = float(close.iloc[-1])

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean().iloc[-1]
        rs = gain / max(loss, 1e-8)
        rsi = 100 - (100 / (1 + rs))

        # Volatilidade
        returns = np.log(close / close.shift(1)).dropna()
        vol_20d = float(returns.rolling(20).std().iloc[-1]) * np.sqrt(252) * 100

        
        if current > sma_20 > sma_50:
            signal = "ALTA (preço acima de SMA20 e SMA50)"
        elif current < sma_20 < sma_50:
            signal = "BAIXA (preço abaixo de SMA20 e SMA50)"
        else:
            signal = "NEUTRO (preço entre SMA20 e SMA50)"

        return (
            f"Análise Técnica: {ticker}\n"
            f"Preço: ${current}\n"
            f"SMA20: ${sma_20} | SMA50: ${sma_50}\n"
            f"RSI(14): {float(rsi)}\n"
            f"Volatilidade anualizada: {vol_20d}%\n"
            f"Sinal: {signal}"
        )
    except Exception as e:
        return f"Erro na análise: {e}"


def _price_prediction(symbol: StockSymbol) -> str:
    """Executa predição de preço usando modelo LSTM treinado."""
    ticker = SYMBOLS_MAP[symbol]

	# Carregar modelo e scaler
    try:
        model, scaler = load_artifacts(symbol)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

	# Carregar histórico de preços
    today = date.today()
    year_before = today - timedelta(days=365)
    df = load_data(ticker, start_date=year_before.isoformat(), end_date=today.isoformat())
    
    if df.empty:
        raise HTTPException(status_code=404, detail='Nenhum dado disponível para o período informado')

	# Pegar apenas coluna de fechamento
    scaled = scaler.transform(df[['Close']].values)

	# Preparar janela de input
    window_size = model.input_shape[1]
    last_window = scaled[-window_size:].reshape(1, window_size, 1)

	# Previsão iterativa
    preds_scaled = []
    for _ in range(5):
        pred = model.predict(last_window, verbose=0)
        preds_scaled.append(pred[0, 0])
        last_window = np.append(last_window[:, 1:, :], [[[pred[0, 0]]]], axis=1)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled)

	# Datas futuras
    last_date = df['Date'].iloc[-1]
    future_dates = [(last_date + timedelta(days=i)).date() for i in range(1, 6)]

	# Salvar previsões em CSV
    file_path = save_predictions_csv(
		symbol=symbol,
		dates=future_dates,
		real_values=None,
		predicted_values=preds.flatten().tolist(),
		model_version='v1',
	)

    return (
            f"Predição para {symbol}:\n"
            f"Predição para os proximo 5 dias: ${preds.flatten().tolist()}\n"
            f"Nota: Para predição quantitativa precisa, use o endpoint /predict da API."
        )


def _rag_knowledge(query: str) -> str:
    """Busca conhecimento relevante do banco de documentos RAG."""
    try:
        contexts = retrieve_context(query, k=3)
        if not contexts:
            return "Nenhum contexto relevante encontrado."
        return "\n---\n".join(contexts)
    except Exception as e:
        logger.error(f"Erro na busca RAG: {e}")
        return f"Erro ao buscar contexto: {e}"
    


def get_stock_tools() -> list[Tool]:
    """Retorna lista de tools para o agente."""
    return [
        Tool(
            name="stock_lookup",
            func=_stock_lookup,
            description=(
                "Consulta dados recentes de uma ação (preço, volume). "
                "Input: Nome da empresa."
            ),
        ),
        Tool(
            name="technical_analysis",
            func=_technical_analysis,
            description=(
                "Calcula indicadores técnicos (SMA, RSI, volatilidade) para uma ação. "
                "Input: Nome da empresa."
            ),
        ),
        Tool(
            name="price_prediction",
            func=_price_prediction,
            description=(
                "Executa predição de preço usando modelo LSTM treinado. "
                "Input: Nome da empresa."
            ),
        ),
        Tool(
            name="rag_knowledge",
            func=_rag_knowledge,
            description=(
                "Busca conhecimento relevante do banco de documentos. "
                "Use para contexto histórico, análises, notícias. "
                "Input: pergunta ou termo de busca."
            ),
        ),
    ]
