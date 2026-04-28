from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from enum import Enum
from datetime import date
import numpy as np
from typing import Optional

from src.ml_utils.inferencia import load_artifacts
from src.ml_utils.train import train_model
from src.api.train_status import TRAIN_STATUS
from src.ml_utils.data_loader import load_data
from src.agente.agente_ia import create_agent
from src.agente.agent_tools import get_tools

from datetime import timedelta
from src.utils.prediction_saver import save_predictions_csv
from src.llm_security.guardrails import InputGuardrail, OutputGuardrail
from src.api.schemas import (
    AgentRequest,
    PredictRequest,
    TrainRequest
)

app = FastAPI(title='TESLA, BYD & TOYOTA LSTM Predictive API')
_INPUT_GUARD = InputGuardrail()
_OUTPUT_GUARD = OutputGuardrail()

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


# ===============================
# ENDPOINT /predict
# ===============================
@app.post('/predict', tags=["Predição"], summary="Previsão de preços futuros usando modelo LSTM")
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
	df = load_data(ticker, start_date=req.start_date.isoformat(), end_date=req.end_date.isoformat())

	if df.empty:
		raise HTTPException(
			status_code=404, detail='Nenhum dado disponível para o período informado'
		)

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
		last_window = np.append(last_window[:, 1:, :], [[[pred[0, 0]]]], axis=1)

	preds_scaled = np.array(preds_scaled).reshape(-1, 1)
	preds = scaler.inverse_transform(preds_scaled)

	# Datas futuras
	last_date = df['Date'].iloc[-1]

	future_dates = [(last_date + timedelta(days=i)).date() for i in range(1, req.days_ahead + 1)]

	# Salvar previsões em CSV
	file_path = save_predictions_csv(
		symbol=symbol,
		dates=future_dates,
		real_values=None,
		predicted_values=preds.flatten().tolist(),
		model_version='v1',
	)

	return {
		'symbol': symbol,
		'ticker': ticker,
		'days_ahead': req.days_ahead,
		'start_date': req.start_date.isoformat(),
		'end_date': req.end_date.isoformat(),
		'predictions': preds.flatten().tolist(),
		'csv_saved_at': str(file_path),
	}


# ===============================
# ENDPOINT /train (background task)
# ===============================
@app.post('/train', tags=["Treinamento"], summary="Treinamento do modelo LSTM")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
	symbol_enum = req.symbol
	symbol = symbol_enum.value
	ticker = SYMBOLS_MAP[symbol_enum]

	# Inicializa status do treino
	TRAIN_STATUS[symbol] = {'progress': 0, 'status': 'Iniciado'}

	# Função em background
	def run_training(symbol, ticker, start_date, end_date, epochs):
		try:
			train_model(
				symbol=symbol,
				ticker=ticker,
				start_date=start_date.isoformat(),
				end_date=end_date.isoformat(),
				epochs=epochs,
			)
			TRAIN_STATUS[symbol]['progress'] = 100
			TRAIN_STATUS[symbol]['status'] = 'Concluído'
		except Exception as e:
			TRAIN_STATUS[symbol]['status'] = f'Erro: {str(e)}'
			TRAIN_STATUS[symbol]['progress'] = 0

	# Executa treino em background
	background_tasks.add_task(
		run_training, symbol, ticker, req.start_date, req.end_date, req.epochs
	)

	return {'message': f'Treinamento de {symbol} iniciado em background'}

# ===============================
# ENDPOINT /LLM Agent
# ===============================
@app.post("/agent", tags=["Agente"], summary="Consulta ao agente ReAct")
async def agent_query(data: AgentRequest) -> AgentResponse:
    REQUEST_COUNT.labels(endpoint="/agent").inc()

    is_valid, reason = _INPUT_GUARD.validate(data.query)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)

    # Lazy import para carregar só se necessário
    tools = get_tools()
    agent = create_agent(tools)
    result = agent.invoke({"input": data.query})

    sanitized = _OUTPUT_GUARD.sanitize(result.get("output", ""))
    return AgentResponse(answer=sanitized)

# ===============================
# STATUS DO TREINO
# ===============================
@app.get('/train/status/{symbol}')
def train_status(symbol: StockSymbol):
	symbol_str = symbol.value
	if symbol_str not in TRAIN_STATUS:
		raise HTTPException(status_code=404, detail='Treino não encontrado')
	return TRAIN_STATUS[symbol_str]


# ===============================
# MONITORAMENTO E AVALIAÇÃO DE QUALIDADE
# ===============================


@app.get("/metrics", tags=["Monitoramento"], summary="Métricas Prometheus")
async def prometheus_metrics() -> JSONResponse:
    return JSONResponse(content=generate_latest().decode(), media_type="text/plain")


@app.post("/evaluate_quality", tags=["Monitoramento"], summary="Avaliar qualidade do modelo")
async def evaluate_quality(data: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from src.monitoring.drift import check_prediction_drift

    y_true = data.get("y_true")
    y_pred_old = data.get("y_pred_old")

    if y_true is None or y_pred_old is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Campos 'y_true' e 'y_pred_old' obrigatórios.",
        )

    y_pred_new = list(_QUALITY_STATE["y_pred_new"])
    if not y_pred_new:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sem predições acumuladas. Faça chamadas /infer antes.",
        )

    sigma_metrics = compute_sigma_metric(
        np.array(y_true), np.array(y_pred_new[: len(y_true)])
    )

    return {"quality_monitoring": sigma_metrics}
