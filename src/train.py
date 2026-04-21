import os
import joblib

# from pathlib import Path
import numpy as np

# import pandas as pd
import random
import tensorflow as tf
from tensorflow import random as tf_random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

from src.data_loader import load_data
from src.preprocessing import create_sequences
from api.train_status import TRAIN_STATUS

import mlflow

MODEL_DIR = 'data/models'
os.makedirs(MODEL_DIR, exist_ok=True)


# ===============================
# CALLBACK DE PROGRESSO
# ===============================
class TrainStatusCallback(Callback):
	def __init__(self, symbol: str):
		self.symbol = symbol

	def on_epoch_end(self, epoch, logs=None):
		if logs is None:
			logs = {}
		TRAIN_STATUS[self.symbol]['progress'] = int(
			(epoch + 1) / TRAIN_STATUS[self.symbol]['epochs'] * 100
		)


# ===============================
# FUNÇÃO PARA SALVAR SCALER
# ===============================
def save_scaler(scaler, symbol: str):
	scaler_path = os.path.join(MODEL_DIR, f'scaler_{symbol}.pkl')
	joblib.dump(scaler, scaler_path)
	mlflow.log_artifact(scaler_path)


# ===============================
# FUNÇÃO DE AVALIAÇÃO
# ===============================
def evaluate(y_true, y_pred):
	mae = np.mean(np.abs(y_true - y_pred))
	rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
	mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
	return {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)}


# ===============================
# CONSTRUÇÃO DO MODELO
# ===============================
def build_model(input_shape, learning_rate: float = 0.001):
	model = Sequential(
		[
			LSTM(50, return_sequences=True, input_shape=input_shape),
			Dropout(0.2),
			LSTM(50),
			Dropout(0.2),
			Dense(1),
		]
	)
	model.compile(optimizer=Adam(learning_rate), loss='mse')

	mlflow.log_param('learning_rate', learning_rate)

	return model


# ===============================
# FUNÇÃO PRINCIPAL DE TREINO
# ===============================
def train_model(
	symbol: str,
	ticker: str,
	epochs: int = 20,
	start_date: str = '2018-01-01',
	end_date: str = '2024-12-31',
	window_size: int = 60,
	batch_size: int = 32,
	train_ratio: float = 0.8,
	early_stopping_patience: int = 15,
	random_seed: int = 42,
):
	# Inicializa status
	TRAIN_STATUS[symbol] = {'status': 'IN_PROGRESS', 'progress': 0, 'epochs': epochs}

	mlflow.set_tracking_uri('file:./mlruns')
	mlflow.set_experiment('LSTM_Training')
	with mlflow.start_run():
		# 1️⃣ Carregar dados
		df = load_data(ticker, start_date, end_date)
		if df.empty:
			TRAIN_STATUS[symbol]['status'] = 'FAILED'
			raise ValueError(
				f'Nenhum dado disponível para {symbol} entre {start_date} e {end_date}'
			)

		# 2️⃣ Pré-processamento
		tf_random.set_seed(random_seed)
		np.random.seed(random_seed)
		random.seed(random_seed)

		X, y, scaler = create_sequences(df, window_size)

		# 3️⃣ Split treino/validação
		split = int(len(X) * train_ratio)
		X_train, X_val = X[:split], X[split:]
		y_train, y_val = y[:split], y[split:]

		# 4️⃣ Construir modelo
		model = build_model((X_train.shape[1], X_train.shape[2]), 0.001)

		# 5️⃣ Callback de progresso
		early_stop = tf.keras.callbacks.EarlyStopping(
			monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True
		)

		callback = [TrainStatusCallback(symbol), early_stop]

		# 6️⃣ Treino e log

		mlflow.log_param('window_size', window_size)
		mlflow.log_param('batch_size', batch_size)
		mlflow.log_param('num_epochs', epochs)
		mlflow.log_param('train_ratio', train_ratio)
		mlflow.log_param('test_ratio', 1 - train_ratio)
		mlflow.log_param('early_stopping_patience', early_stopping_patience)
		mlflow.log_param('random_seed', random_seed)

		history = model.fit(
			X_train,
			y_train,
			validation_data=(X_val, y_val),
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callback,
			verbose=1,
		)

		for epoch in range(len(history.history['loss'])):
			mlflow.log_metric('train_loss', history.history['loss'][epoch], step=epoch)
			mlflow.log_metric('val_loss', history.history['val_loss'][epoch], step=epoch)

		# 7️⃣ Avaliação
		preds = model.predict(X_val)
		metrics = evaluate(y_val, preds)

		mlflow.log_metric('val_mae', metrics['mae'])
		mlflow.log_metric('val_rmse', metrics['rmse'])
		mlflow.log_metric('val_mape', metrics['mape'])

		# 8️⃣ Salvar modelo e scaler
		result = save_model(model, scaler, symbol, metrics, epochs)

		return result


def save_model(
	model,
	scaler,
	symbol: str,
	metrics: dict,
	epochs: int,
):
		# 8️⃣ Salvar modelo e scaler
		model_path = os.path.join(MODEL_DIR, f'lstm_{symbol}.keras')
		model.save(model_path)
		save_scaler(scaler, symbol)
		mlflow.log_artifact(model_path)

		# Atualiza status final
		TRAIN_STATUS[symbol]['status'] = 'COMPLETED'
		TRAIN_STATUS[symbol]['progress'] = 100
		TRAIN_STATUS[symbol]['metrics'] = metrics
		TRAIN_STATUS[symbol]['model_path'] = model_path

		return {'symbol': symbol, 'epochs': epochs, 'model_path': model_path, 'metrics': metrics}


if __name__ == '__main__':
	results = []
	for name, ticker in SYMBOLS.items():
		results.append(train_symbol(name, ticker))

	print('Treinamento finalizado:')
	print(results)
