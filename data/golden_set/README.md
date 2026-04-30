# Golden Set para Avaliação do Agente ReAct

## Descrição

Golden set é um dataset de **query + expected_answer + contexts** usado para avaliar qualidade do agente ReAct através de métricas RAGAS (Retrieval-Augmented Generation Assessment).

## Estrutura do JSON

```json
{
  "query": "Pergunta do usuário final",
  "expected_answer": "Resposta esperada (gold standard)",
  "contexts": [
    "Contexto 1 (documento recuperado relevante)",
    "Contexto 2 (documento recuperado relevante)",
    "Contexto 3 (documento recuperado relevante)"
  ]
}
```

### Campo `query`
- Pergunta natural que um usuário faria sobre análise de ações
- Deve ser realista e ter resposta bem-definida
- Exemplos: "Qual a tendência da Tesla?", "TSLA caiu 5% hoje. Devo comprar?"

### Campo `expected_answer`
- Resposta ideal esperada (não necessariamente única)
- Deve incluir dados técnicos (SMA, RSI, volume) ou fundamentais (P/E, dividendos)
- Pode ter caveats ("depende do cenário técnico", "disclaimer")
- ~100-200 palavras típicamente

### Campo `contexts`
- Lista de 2-4 documentos contextuais relevantes
- Simula output de RAG retriever
- Deve conter informações necessárias para responder a query
- Cada context é ~1-3 sentenças

---

## Métricas RAGAS Calculadas

Com este golden set, você pode avaliar:

| Métrica | O que mede | Target |
|---------|-----------|--------|
| **Faithfulness** | % resposta baseada nos contexts | ≥0.80 |
| **Answer Relevancy** | % resposta relevante à query | ≥0.85 |
| **Context Precision** | % contexts úteis (não ruído) | ≥0.80 |
| **Context Recall** | % contexts necessários foram retornados | ≥0.75 |

## Como Usar

### 1. Validar Golden Set Localmente

```python
import json
import pandas as pd

with open("data/golden_set/golden_set.json") as f:
    golden_set = json.load(f)

# Verificar estrutura
for i, item in enumerate(golden_set):
    assert "query" in item, f"Item {i} sem query"
    assert "expected_answer" in item, f"Item {i} sem expected_answer"
    assert "contexts" in item, f"Item {i} sem contexts"
    assert len(item["contexts"]) >= 2, f"Item {i} com <2 contexts"

print(f"✓ {len(golden_set)} pares validados")
```

### 2. Rodar Avaliação RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Carregar golden set
with open("data/golden_set/golden_set.json") as f:
    items = json.load(f)

# Gerar respostas do agente
results = []
for item in items:
    answer, contexts = agent.invoke({"input": item["query"]})
    results.append({
        "question": item["query"],
        "answer": answer,
        "contexts": contexts,
        "ground_truth": item["expected_answer"]
    })

# Avaliar
dataset = Dataset.from_list(results)
scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(f"Faithfulness: {scores['faithfulness']:.3f}")
print(f"Answer Relevancy: {scores['answer_relevancy']:.3f}")
print(f"Context Precision: {scores['context_precision']:.3f}")
print(f"Context Recall: {scores['context_recall']:.3f}")
```

### 3. Usar em CI/CD

```yaml
# .github/workflows/evaluation.yml
jobs:
  ragas-eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: RAGAS Evaluation
        run: python evaluation/ragas_eval.py data/golden_set/golden_set.json
```

---

## Cobertura de Tópicos (20 pares)

| # | Tópico | Cobertura |
|---|--------|-----------|
| 1-3 | Análise de tendência (TSLA, BYD, TOYOTA) | 3 queries |
| 4-6 | Indicadores técnicos (volatilidade, RSI, SMA) | 3 queries |
| 7-9 | Estratégia de trading (buy/sell, dips) | 3 queries |
| 10-12 | Análise fundamental (P/E, dividendos, earnings) | 3 queries |
| 13-15 | Risco e diversificação | 3 queries |
| 16-18 | Modelos e metodologia (LSTM, técnica vs fundamental) | 3 queries |
| 19-20 | Limitações e caveats | 2 queries |

---

## Padrões de Qualidade

### ✓ Bom Golden Set
```json
{
  "query": "Como identificar uma bolha especulativa?",
  "expected_answer": "Sinais: P/E extremamente alto (>60x), volume especulativo, narrativa FOMO...",
  "contexts": [
    "Bolha: quando preço desconectado de valor intrínseco. Exemplos: Dot-com (2000)...",
    "TSLA P/E alto (35x) pode ser bolha OU justified se crescimento 30%/ano..."
  ]
}
```

### ✗ Ruim Golden Set
```json
{
  "query": "Tesla sobe?",  // vago, sem contexto
  "expected_answer": "sim",  // resposta muito curta
  "contexts": ["Tesla"]  // contexto não informativo
}
```

---

## Manutenção

### Quando Atualizar
- Mensalmente: adicionar novas queries sobre eventos recentes
- Trimestralmente: revisar accurácia das expected_answers com dados reais
- Após major mudanças: validar golden set ainda é representativo

### Versionamento
Usar branches para versões:
```bash
git checkout -b golden-set/v2-expanded
# Adicionar 10+ queries novas
git commit -m "feat: expand golden set to 30 queries"
```

---

## Disclaimer

Este golden set é **ilustrativo e educacional**. Não é recomendação de investimento. Respostas refletem análise técnica/fundamental simplificada; decisões reais requerem pesquisa aprofundada e orientação de profissionais.

---

**Criado em:** Abril 2026  
**Versão:** 1.0  
**Total de pares:** 20  
**Ativos cobertos:** TSLA, BYD, TOYOTA (TM)
