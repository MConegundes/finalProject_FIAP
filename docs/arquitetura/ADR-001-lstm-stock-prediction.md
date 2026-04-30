# ADR-001: LSTM para Previsão de Preços de Ações

## Problema

Prever preços de fechamento de ações com tolerância de erro ≤ 0.5σ (desvio-padrão) em pelo menos 70% das predições.

## Decisão

Usar **LSTM (Long Short-Term Memory)** em PyTorch como modelo base para capturar dependências temporais em séries de preços.

### Justificativa

1. **Série Temporal:** Preços de ações exibem dependências de longo prazo que LSTM captura naturalmente
2. **Flexibilidade:** Arquitetura permitindo stacking de camadas e dropout para regularização
3. **Produção:** Serialização em `.keras` ou `.pt` com inferência rápida
4. **Benchmarking:** Baseline Ridge (estatístico) é 2-3x mais rápido mas com menor acurácia

### Arquitetura Proposta

```
┌──────────────────┐
│  Preços (Close)  │  Shape: (n, 1)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ MinMaxScaler     │  Normalização [0, 1]
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Janelas Sequenciais (window=60)     │  Shape: (n_samples, 60, 1)
│  input[i] = preços[i:i+60]           │
│  target[i] = preço[i+60]             │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  LSTM (64 units, ReLU)   │  Shape: (n, 64)
│  Dropout(0.2)            │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  LSTM (32 units, ReLU)   │  Shape: (n, 32)
│  Dropout(0.2)            │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Dense(1, Linear)        │  Shape: (n, 1)
│  Output (previsão)       │
└──────────────────────────┘
```

### Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `window_size` | 60 dias | ~3 meses de histórico; captura tendências sazonais |
| `lstm_units_1` | 64 | Balança capacidade e overfitting |
| `lstm_units_2` | 32 | Redução progressiva |
| `dropout` | 0.2 | Regularização moderada |
| `batch_size` | 32 | Trade-off entre memória e convergência |
| `epochs` | 100 | Com early stopping em validation loss |
| `optimizer` | Adam | Learning rate adaptativo |
| `loss` | MSE | Apropriado para regressão |

### Normalização

- **MinMaxScaler:** Transforma valores para [0, 1] antes do treinamento
- **Desnormalização:** Aplicada post-predição para retornar valores reais
- **Armazenamento:** Scaler salvo em `data/models/{symbol}_scaler.save` via `joblib`

### Treinamento

1. **Split temporal:** 80% treino, 20% validação
2. **Early stopping:** Se validation loss não melhorar por 10 épocas, para
3. **Métrica:** MAE, RMSE e σ-error (% dentro de 0.5σ)
4. **Logging:** Todas as métricas via MLflow com tags padronizadas

## Consequências

### Positivas
✅ Captura dependências temporais  
✅ Escalável para múltiplos ativos  
✅ Fácil integração em API FastAPI  
✅ Suporta quantização (INT4) para inference rápido  

### Negativas
❌ Requer GPU para treino rápido  
❌ Propenso a overfitting em dados limitados  
❌ Hyperparameter tuning crítico  

## Mitigação

- ✓ Dropout + L2 regularization contra overfitting
- ✓ Validação cruzada temporal (não shuffled)
- ✓ Métricas de drift monitoradas continuamente

