import logging

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logger = logging.getLogger(__name__)

def check_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> dict:
    """Args:
        reference_data: DataFrame de treinamento (referência).
        current_data: DataFrame de produção (atual).
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    result = report.as_dict()
    drift_result = result["metrics"][0]["result"]

    share_drifted = drift_result.get("share_of_drifted_columns", 0)
    is_drift = drift_result.get("dataset_drift", False)

    return {
        "share_of_drifted_columns": share_drifted,
        "dataset_drift": is_drift,
        "number_of_columns": drift_result.get("number_of_columns", 0),
        "number_of_drifted_columns": drift_result.get("number_of_drifted_columns", 0),
    }


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Args:
        reference: Distribuição de referência.
        current: Distribuição atual.
        n_bins: Número de bins para discretização.

    Returns:
        Valor PSI (0 = sem drift, >0.2 = drift significativo).
    """
    eps = 1e-4

    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

    ref_counts = np.clip(ref_counts, eps, None)
    cur_counts = np.clip(cur_counts, eps, None)

    psi = float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))

    return psi


def check_prediction_drift(
    y_pred_reference: np.ndarray,
    y_pred_current: np.ndarray,
    psi_warning: float = 0.1,
    psi_critical: float = 0.2,
) -> dict:
    """Args:
        y_pred_reference: Predições de referência (treino/validação).
        y_pred_current: Predições atuais (produção).
        psi_warning: Threshold de warning.
        psi_critical: Threshold de retrain.

    Returns:
        Dicionário com PSI e status.
    """
    psi = compute_psi(y_pred_reference, y_pred_current)

    status = "ok"
    if psi > psi_critical:
        status = "critical"
        logger.warning("DRIFT CRÍTICO: PSI=%.4f > %.2f → retrain recomendado", psi, psi_critical)
    elif psi > psi_warning:
        status = "warning"
        logger.warning("DRIFT WARNING: PSI=%.4f > %.2f", psi, psi_warning)

    return {"psi": psi, "status": status, "thresholds": {"warning": psi_warning, "critical": psi_critical}}
