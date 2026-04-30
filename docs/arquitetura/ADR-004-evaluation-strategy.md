# ADR-004: Estratégia de Avaliação (LSTM + Agente)

## Problema

Avaliar simultaneamente:
1. **Modelo LSTM:** Acurácia em previsão de preços
2. **Agente ReAct:** Qualidade de respostas financeiras
3. **Sistema Ponta-a-Ponta:** Integração LSTM + Agente + Guardrails

Sem métricas claras, não há como demonstrar maturidade produtiva.

## Decisão

Implementar **avaliação em 3 camadas**:
1. **LSTM (regressão):** MAE, RMSE, σ-error
2. **Agente RAG (retrieval-generation):** RAGAS 4 métricas
3. **Sistema (human judgment):** LLM-as-judge com 3+ critérios

### 1. Avaliação de Regressão (LSTM)

#### Métricas Principais

| Métrica | Cálculo | Target | Interpretação |
|---------|---------|--------|----------------|
| **MAE** | Σ\|pred - real\| / n | <2% do preço | Erro médio absoluto |
| **RMSE** | √(Σ(pred-real)² / n) | <3% do preço | Penaliza outliers |
| **σ-error** | % predições ≤ 0.5σ | ≥70% | **Métrica de negócio** |
| **MAPE** | Σ\|pred-real\|/real / n | <5% | Erro percentual médio |

#### Estratégia de Validação Temporal

```python
# NÃO usar shuffle em séries temporais!
# Validação Walk-Forward (backtest)

def walk_forward_validation(df, window=60, test_size=30):
    results = []
    for t in range(len(df) - window - test_size):
        train_data = df[t:t+window]
        test_data = df[t+window:t+window+test_size]
        
        model.train(train_data)
        preds = model.predict(test_data)
        metrics = compute_metrics(preds, test_data)
        results.append(metrics)
    
    return aggregate_results(results)
```

#### Baseline de Comparação

| Modelo | MAE | σ-error |
|--------|-----|---------|
| Naive (último valor) | 2.5% | 45% |
| Ridge Regression | 1.8% | 58% |
| **LSTM Proposto** | **<1.5%** | **≥70%** |

---

### 2. Avaliação de Agente RAG (RAGAS)

#### Golden Set (≥20 pares)

Estrutura JSON:
```json
{
  "query": "Qual a tendência da Tesla?",
  "expected_answer": "Tesla está em alta...",
  "contexts": [
    "Tesla reportou lucro record em 2024...",
    "Analistas sugerem buy..."
  ]
}
```

#### 4 Métricas RAGAS (Obrigatório)

| Métrica | O que mede | Target |
|---------|-----------|--------|
| **Faithfulness** | % info da resposta baseada nos contextos | ≥0.80 |
| **Answer Relevancy** | % resposta é relevante à pergunta | ≥0.85 |
| **Context Precision** | % contextos úteis (não ruído) | ≥0.80 |
| **Context Recall** | % contextos necessários foram retornados | ≥0.75 |

#### Implementação

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall
)

scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy,
             context_precision, context_recall]
)

print(f"Faithfulness: {scores['faithfulness']:.3f}")
print(f"Answer Relevancy: {scores['answer_relevancy']:.3f}")
```

---

### 3. Avaliação Humana (LLM-as-Judge)

#### Critérios de Avaliação (≥3)

```python
judge_template = """
Avalie a resposta do agente financeiro de 0-10:

Pergunta: {query}
Resposta: {answer}

Critério 1 - Acurácia Técnica (0-10):
- Indicadores técnicos mencionados são corretos?
- Previsões condizem com modelo LSTM?

Critério 2 - Relevância para Negócio (0-10):
- A resposta ajuda o trader a tomar decisão?
- Contexto é importante para o caso de uso?

Critério 3 - Segurança/Responsabilidade (0-10):
- Evita garantias falsas ("100% vai subir")?
- Menciona riscos?

Score final = (C1 + C2 + C3) / 3
"""
```

#### Execução

```python
from langchain.llms import ChatOpenAI

judge = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
score = judge.predict(prompt=judge_template.format(
    query=q, answer=agent_response
))
```

---

## Estratégia Integrada

```
┌─────────────────────────────┐
│  Query: "Compro TSLA?"      │
└──────────┬──────────────────┘
           │
           ▼
    ┌──────────────────────┐
    │  Agente ReAct        │
    │  1. stock_lookup     │
    │  2. technical_analysis
    │  3. price_prediction │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Output Guardrails       │
    │  (Remove PII)            │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Response                │
    │  "TSLA em alta, RSI=65"  │
    └──────────┬───────────────┘
               │
               ├─► RAGAS 4 métricas ✓
               │   Faithfulness: 0.85
               │
               ├─► LLM Judge 3 critérios ✓
               │   Accuracy: 8/10
               │   Relevance: 9/10
               │   Safety: 9/10
               │
               └─► Prometheus metrics ✓
                   latency_p95: 2.3s
                   tool_success: 100%
```

---

## Artefatos

| Arquivo | Conteúdo | Frequência |
|---------|----------|-----------|
| `evaluation/ragas_eval.py` | Script RAGAS | Por release |
| `evaluation/llm_judge.py` | Script Judge | Por release |
| `data/golden_set.json` | 20+ pares | Mantém-se |
| `data/test_predictions/` | Outputs para validação | Por execução |

---

## Thresholds de Aceite

| Componente | Métrica | Target | Status |
|------------|---------|--------|--------|
| **LSTM** | σ-error | ≥70% | ✓ Obrigatório |
| **LSTM** | MAE | <2% | ✓ Obrigatório |
| **Agente** | Faithfulness | ≥0.80 | ✓ Obrigatório |
| **Agente** | Answer Relevancy | ≥0.85 | ✓ Obrigatório |
| **Agente** | Latency P95 | <5s | ✓ Recomendado |
| **Judge** | Score médio | ≥8.0/10 | ✓ Recomendado |

---

## Consequências

### Positivas
✅ Múltiplas perspectivas (técnica + negócio + humano)  
✅ Reprodutível e documentado  
✅ Alinhado com rubrica Datathon  
✅ Detecta degradação rapidamente  

### Negativas
❌ Custo computacional (RAGAS + judge)  
❌ Golden set requer curadoria manual  
❌ Métricas nem sempre concordam  
