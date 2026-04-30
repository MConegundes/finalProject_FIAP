"""metrics.py

Métricas Prometheus para observabilidade da API.
"""
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "datathon_requests_total",
    "Total de requisições à API",
    ["endpoint"],
)

INFERENCE_LATENCY = Histogram(
    "datathon_inference_latency_seconds",
    "Latência de inferência em segundos",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)