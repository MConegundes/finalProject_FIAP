from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from typing import Optional
from datetime import date
from enum import Enum


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
# SCHEMAS
# ===============================
class PredictRequest(BaseModel):
	"""Requisição para predição de preços futuros usando modelo LSTM."""
	symbol: StockSymbol
	days_ahead: int = Field(default=5, ge=1, le=60)
	start_date: date = Field(default=date(2015, 1, 1))
	end_date: date = Field(default=date(2025, 12, 31))

	@field_validator('end_date')
	@classmethod
	def check_dates(cls, v: date, info: ValidationInfo):
		start_date = info.data.get('start_date')

		if start_date and v < start_date:
			raise ValueError('A data de término deve ser maior ou igual à data de início.')

		return v


class TrainRequest(BaseModel):
	"""Requisição para treinamento do modelo LSTM."""
	symbol: StockSymbol
	start_date: Optional[date] = Field(default=date(2015, 1, 1))
	end_date: Optional[date] = Field(default=date(2025, 12, 31))
	epochs: Optional[int] = Field(default=20, ge=1, le=100)
	

class AgentRequest(BaseModel):
    """Requisição para o agente ReAct."""

    query: str = Field(min_length=1, max_length=4096, description="Pergunta do usuário")
    include_rag: bool = Field(default=True, description="Incluir contexto RAG")
