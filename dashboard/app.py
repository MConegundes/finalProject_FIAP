import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
st.set_page_config(page_title='LSTM – Monitoramento', layout='wide')
st.title('📊 Prediction (TESLA, BYD and TOYOTA) – Monitoramento em Produção')

# -------------------------------------------------------------------
# ENV
# -------------------------------------------------------------------
load_dotenv('configure.env')
PREDICTIONS = os.getenv('PREDICTIONS_DIR')
PREDICTIONS_DIR = Path(PREDICTIONS)


# =========================
# SELEÇÃO
# =========================
st.header('🔮 Previsões Geradas')

symbol = st.selectbox('Empresa', ['TSLA', 'BYD', 'TOYOTA'])

# =========================
# VERIFICA CSVs
# =========================
if not PREDICTIONS_DIR.exists():
	st.warning('📂 Diretório data/predictions não encontrado.')
	st.stop()

csv_files = sorted(
	PREDICTIONS_DIR.glob(f'{symbol}_*.csv'), key=lambda x: x.stat().st_mtime, reverse=True
)

if not csv_files:
	st.warning(f'⚠️ Nenhuma previsão encontrada para {symbol}.')
	st.stop()

latest_file = csv_files[0]

st.success(f'📄 Usando arquivo: `{latest_file.name}`')

# =========================
# LEITURA DO CSV
# =========================
df = pd.read_csv(latest_file, parse_dates=['date'])

df = df.sort_values('date')

# =========================
# GRÁFICO
# =========================
st.subheader('📈 Previsão de Preço')

fig, ax = plt.subplots(figsize=(10, 5))

if 'real' in df.columns and df['real'].notna().any():
	ax.plot(df['date'], df['real'], label='Real', linewidth=2)

ax.plot(df['date'], df['predicted'], label='Previsto', linestyle='--')

ax.set_xlabel('Data')
ax.set_ylabel('Preço')
ax.legend()
ax.grid(True)

st.pyplot(fig)

# =========================
# TABELA
# =========================
with st.expander('📋 Dados da Previsão'):
	st.dataframe(df)


def load_latest_predictions():
	base_dir = Path(__file__).resolve().parents[1]
	#    predictions_dir = base_dir / "artifacts" / "predictions"
	predictions_dir = base_dir / 'data' / 'predictions'

	if not predictions_dir.exists():
		return None, 'Pasta data/predictions não encontrada'

	csv_files = list(predictions_dir.glob('*.csv'))

	if not csv_files:
		return None, 'Nenhum arquivo de previsão encontrado'

	latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
	df = pd.read_csv(latest_file)

	return df, latest_file.name


# =========================
# Latest Predictions Section
# =========================
st.header('🗂️ Última Previsão Gerada')

df_pred, info = load_latest_predictions()

if df_pred is None:
	st.info(info)
else:
	st.success(f'Arquivo carregado: {info}')

	df_pred['date'] = pd.to_datetime(df_pred['date'])

	st.line_chart(df_pred.set_index('date')[['predicted']])

	with st.expander('📄 Visualizar dados'):
		st.dataframe(df_pred)


# =========================
# MULTI-EMPRESA – GRÁFICO CONSOLIDADO
# =========================
st.header('📊 Comparativo de Previsões – Todas as Empresas')

symbols = ['TSLA', 'BYD', 'TOYOTA']

fig_all, ax_all = plt.subplots(figsize=(12, 6))

found_any = False

for symbol in symbols:
	csv_files = sorted(
		PREDICTIONS_DIR.glob(f'{symbol}_*.csv'), key=lambda x: x.stat().st_mtime, reverse=True
	)

	if not csv_files:
		continue

	latest_file = csv_files[0]
	df_sym = pd.read_csv(latest_file, parse_dates=['date'])
	df_sym = df_sym.sort_values('date')

	ax_all.plot(df_sym['date'], df_sym['predicted'], label=symbol, linewidth=2)

	found_any = True

if not found_any:
	st.warning('⚠️ Nenhuma previsão encontrada para nenhuma empresa.')
else:
	ax_all.set_xlabel('Data')
	ax_all.set_ylabel('Preço Previsto')
	ax_all.set_title('Previsões mais recentes por empresa')
	ax_all.legend()
	ax_all.grid(True)

	st.pyplot(fig_all)
