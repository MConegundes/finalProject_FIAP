from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total de requisições",
    ["endpoint", "method", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_latency_seconds",
    "Latência das requisições",
    ["endpoint"]
)

MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Total de predições feitas",
    ["symbol"]
)

MODEL_ERRORS = Counter(
    "model_errors_total",
    "Total de erros do modelo",
    ["symbol"]
)