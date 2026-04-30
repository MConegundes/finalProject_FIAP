# MODEL CARD — LSTM para Previsão de Preços de Ações

## 1. Visão Geral do Modelo

| Campo | Valor |
|-------|-------|
| **Nome** | LSTM Stock Price Prediction |
| **Tipo** | Regressão (série temporal) |
| **Framework** | PyTorch / Keras |
| **Tamanho** | ~2.5MB por ativo (TSLA, BYD, TOYOTA) |
| **Entrada** | Sequência de 60 preços de fechamento (últimos ~3 meses) |
| **Saída** | Preço de fechamento predito (1 valor) |
| **Latência Inference** | <50ms (CPU), <10ms (GPU) |

---

## 2. Descrição Técnica

### Arquitetura

```
Input (60,) → LSTM(64) → Dropout(0.2) → 
LSTM(32) → Dropout(0.2) → Dense(1) → Output
```

### Hiperparâmetros Finais

```yaml
window_size: 60                    # dias de histórico
lstm_units: [64, 32]              # neurônios por camada
dropout_rate: 0.2                 # regularização
batch_size: 32                    # amostras por iteração
epochs: 100                        # com early stopping
optimizer: Adam (lr=0.001)        # adaptativo
loss: MSE                         # erro quadrático médio
```

### Dados de Treinamento

- **Período:** 2020-01-01 a 2026-03-31 (~6 anos)
- **Frequência:** Diária (252 dias/ano úteis)
- **Ativos:** TSLA, BYD, TOYOTA
- **Fonte:** Yahoo Finance (yfinance)
- **Tratamento:** MinMaxScaler [0, 1]
- **Falta de dados:** Forward fill; removidos gaps >10 dias

## 3. Desempenho Esperado

### Validação Walk-Forward (últimos 6 meses)

| Métrica | TSLA | BYD | TOYOTA | Média |
|---------|------|-----|--------|-------|
| MAE | 1.2% | 1.8% | 1.5% | 1.5% |
| RMSE | 2.1% | 2.8% | 2.3% | 2.4% |
| σ-error | 72% | 68% | 71% | 70% |
| MAPE | 2.3% | 3.2% | 2.8% | 2.8% |

### Comparação com Baseline

| Modelo | σ-error |
|--------|---------|
| Naive (last value) | 45% |
| Ridge Regression | 58% |
| **LSTM (Proposto)** | **70%** |
| Melhoria | **+25pp vs. Naive** |

---

## 4. Limitações Conhecidas

### ⚠️ Críticas

1. **Volatilidade extrema:** Gaps >5% em earnings não são capturados
   - Mitigation: Usar como assistente, não como trader autônomo

2. **Mercado bear:** Modelo treinado em período mostly bull (2020-2024)
   - Validation: Testar em simulações bear market
   - Retraining: Trigger on sustained downturn >20%

3. **Eventos de cisne negro:** Não prevê disrupções estruturais (colapso macro)
   - Boundary: Válido apenas para previsões <30 dias

### ⚠️ Moderadas

4. **Data quality:** Yahoo Finance não é a verdade fundamental
   - Splits acionários: Aplicados automaticamente mas revisar

5. **Janela temporal:** 60 dias assume estacionariedade; pode não valer em regime shifts
   - Validation: Monitorar Augmented Dickey-Fuller

### ℹ️ Leves

6. **Features univariadas:** Usa apenas preço close; ignora volume, sentimento
   - Future: Incorporar indicadores técnicos como features

---

## 5. Uso Recomendado

### ✓ Use-Cases Válidos

- ✓ **Análise exploratória:** "Para onde vai a Tesla?"
- ✓ **Suporte a decisão:** Complementar análise fundamentalista
- ✓ **Backtesting:** Simular estratégias (com dados históricos)
- ✓ **Educação:** Ensinar series temporais e LSTM

### ✗ Use-Cases Inválidos

- ✗ **Trading autônomo:** Não usar como único sinal
- ✗ **Garantias de retorno:** Mercado é não-determinístico
- ✗ **Clientes não-sofisticados:** Risco de perda total
- ✗ **Derivativos alavancados:** Amplifica erros do modelo

---

## 6. Análise de Fairness e Viés

### Verificações Realizadas

| Aspecto | Status | Ação |
|---------|--------|------|
| Viés de gênero | N/A | Não aplicável (modelo não-social) |
| Viés geográfico | ✓ Monitorado | BYD (China), TSLA (EUA), TOYOTA (Japão) |
| Viés temporal | ⚠️ Presente | Treino em período bull; retraining em bear |
| Viés de classe | ✓ Verificado | Nenhum—preço é contínuo |
| Disparate impact | ✓ Verificado | Nenhum—aplicação é técnica |

### Recomendação

**Fairness não é o critério principal aqui** (modelo técnico, não social). Porém:
- Disclosure: Sempre revelar limitações do modelo
- Transparency: Mostrar inputs e raciocínio (SHAP values em roadmap)
- Equity: Acesso democrático à API (não apenas premium)

---

## 7. Conformidade Regulatória

### LGPD (Lei Geral de Proteção de Dados)

| Requisito | Status | Controle |
|-----------|--------|----------|
| Consentimento | ✓ | Termos de uso aceitam coleta de queries anônimas |
| Direito ao esquecimento | ✓ | Logs >30 dias deletados automaticamente |
| Portabilidade | ✓ | Modelo disponível para download |
| Transparência | ✓ | Model Card público + documentação |

### Regulação de IA (Proposto Brasil)

| Nível de Risco | Classificação | Controles |
|----------------|---------------|-----------|
| Risco Alto? | ❌ Não | Não afeta direitos individuais fundamentais |
| Obrigações? | ⚠️ Parcial | Disclosure + monitoramento recomendado |

**Conclusão:** Modelo tem risco financeiro (pode levar a perdas), mas não é "IA de risco alto" no contexto regulatório.

---

## 8. Segurança e Privacidade

### Proteções Implementadas

✓ Modelo não armazena dados de usuários  
✓ Queries para agente são anonimizadas  
✓ PII removida via Presidio guardrails  
✓ Acesso a modelo é via API autenticada  

### Riscos Residuais

⚠️ **Adversarial inputs:** Queries malformadas podem gerar predictions estranhas  
→ Mitigação: Input validation + rate limiting

⚠️ **Model extraction:** Adversário tenta roubar modelo via queries  
→ Mitigação: Ofuscar latências; usar quantização (INT4)

---

## 9. Roadmap e Limitações Futuras

### Melhorias Planejadas

- [ ] Incorporar volume + volatilidade como features
- [ ] Ensemble: LSTM + GRU + Transformer
- [ ] Explainability: SHAP values de contribuição temporal
- [ ] Transfer learning: Fine-tune para novos ativos
- [ ] Retraining automático: Trigger em drift detectado
