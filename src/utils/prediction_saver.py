from pathlib import Path
import pandas as pd
from datetime import date, datetime
from typing import List, Optional

# Resolve a raiz do projeto
BASE_DIR = Path(__file__).resolve().parents[2]

# Pasta de saída
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def save_predictions_csv(
    symbol: str,
    dates: List[date],
    predicted_values: List[float],
    real_values: Optional[List[float]] = None,
    model_version: str = "v1"
) -> Path:
    """
    Salva previsões em CSV para auditoria e monitoramento.

    Formato do arquivo:
    TSLA_predictions_v1_YYYYMMDD_HHMMSS.csv
    """

    if real_values is None:
        real_values = [None] * len(predicted_values)

    df = pd.DataFrame({
        "date": dates,
        "real": real_values,
        "predicted": predicted_values
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_name = f"{symbol}_predictions_{model_version}_{timestamp}.csv"

    file_path = PREDICTIONS_DIR / file_name
    df.to_csv(file_path, index=False)

    return file_path


