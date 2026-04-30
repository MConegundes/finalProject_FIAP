# SYSTEM CARD — Arquitetura Segura do Sistema LSTM + Agente

**Versão:** 1.0  
**Data:** Abril 2026  
**Escopo:** Visão integrada de segurança, governança e operações

---

## 1. Componentes do Sistema

```
┌─────────────────────────────────────────────────────┐
│                   Cliente (Web/API)                 │
└────────┬────────────────────────────────────────────┘
         │ HTTPS + Auth Token
         ▼
┌─────────────────────────────────────────────────────┐
│        FastAPI Gateway (Secrets Vault)              │
│  ├─ Input Guardrails (Prompt Injection Detection)   │
│  ├─ Rate Limiting (10 req/min por user)             │
│  └─ Request Validation (Pydantic schemas)           │
└────┬────────────────┬────────────────┬──────────────┘
     │                │                │
     ▼                ▼                ▼
┌──────────┐   ┌─────────────┐   ┌──────────────┐
│   LSTM   │   │   Agente    │   │   RAG/ChromaDB
│          │   │   (Qwen)    │   │   Embeddings
│ PyTorch  │   │  INT4 NF4   │   │
└──────────┘   └──────┬──────┘   └──────────────┘
                      │
                      ▼ Tools
                ┌────────────────┐
                │ yfinance       │
                │ calculations   │
                │ predictions    │
                └────────────────┘
                      │
     ┌────────────────┴────────────────┐
     │                                  │
     ▼                                  ▼
┌──────────────┐              ┌──────────────────┐
│ Output       │              │ MLflow Registry  │
│ Guardrails   │              │ (Model versions) │
│ (PII removal)│              │ + Metrics        │
└──────────────┘              └──────────────────┘

┌──────────────────────────────────────────────┐
│  Observabilidade                             │
│  ├─ Prometheus (métricas)                    │
│  ├─ Grafana (dashboards)                     │
│  ├─ Evidently (drift detection)              │
│  └─ Logs estruturados (ELK)                  │
└──────────────────────────────────────────────┘
```

---

## 2. Fluxo de Segurança por Camada

### Camada 1: API Gateway (FastAPI)

**Controles:**
- ✓ TLS 1.3 obrigatório (HTTPS)
- ✓ Autenticação: JWT tokens com exp 24h
- ✓ Rate limiting: 10 req/min por IP
- ✓ CORS: Apenas domínios whitelisted

**Código:**
```python
from fastapi import FastAPI, Depends, HTTPException

### Camada 2: Input Validation & Guardrails

**Ameaças Mitigadas:**
- ✗ Prompt injection (ex: "ignore previous instructions")
- ✗ Context stuffing (queries >4096 chars)

**Implementação:**
```python
class InputGuardrail:
    INJECTION_PATTERNS = [
        r"ignore.*previous.*instructions",
        r"you.*are.*now.*a",
        r"<\|im_start\|>",  # LLaMA tokens
        r"\[INST\]",        # Mistral tokens
    ]
    
    def validate(self, user_input: str) -> (bool, str):
        # Check 1: Injection patterns
        # Check 2: Length (<4096)
        return is_valid, reason
```

---

### Camada 3: LLM Execution (Qwen INT4)

**Controles:**
- ✓ Quantização: INT4 reduz surface de ataque (menos parâmetros expostos)
- ✓ Temperatura: 0.3 (determinístico, evita "criatividade" adversarial)
- ✓ Max tokens: 512 (evita loops infinitos)

**Código:**
```python
generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.3,      # Conservador
    "do_sample": False,      # Determinístico
}

# Com timeout
import signal
def timeout_handler(signum, frame):
    raise TimeoutError("LLM inference timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30s max
try:
    response = model.generate(...)
finally:
    signal.alarm(0)
```

---

### Camada 4: Tools Execution

**Segurança de Tools:**

| Tool | Risco | Controle |
|------|-------|----------|
| `stock_lookup` | Nulo | yfinance é somente-leitura |
| `technical_analysis` | Nulo | Cálculos locais |
| `price_prediction` | Baixo | Modelo determinístico; sem I/O externo |
| `rag_knowledge` | Médio | ChromaDB local; sem queries SQL |


### Camada 5: Output Sanitization

**Ameaças Mitigadas:**
- ✗ **Jailbreak Feedback:** Output contém instruções maliciosas
- ✗ **Statistical Disclosure:** Agregados revelam informações privadas


## 3. Matriz de Ameaças x Controles

| # | Ameaça | OWASP LLM | Severidade | Controle | Status |
|---|--------|-----------|-----------|----------|--------|
| 1 | Prompt Injection | LLM01 | Alto | InputGuardrail regex | ✓ Implementado |
| 2 | Insecure I/O Handling | LLM02 | Alto | Output sanitization (Presidio) | ✓ Implementado |
| 3 | Training Data Poisoning | LLM03 | Alto | Dados versionados (DVC) | ✓ Implementado |
| 4 | Model Denial of Service | LLM04 | Médio | Rate limiting + timeouts | ✓ Implementado |
| 5 | Supply Chain Vulnerabilities | LLM05 | Médio | Dependency scanning (pip-audit) | ✓ Implementado |
| 6 | Sensitive Information Disclosure | LLM06 | Alto | Logging sanitizado; secrets em .env | ✓ Implementado |
| 7 | Insecure Plugin Design | LLM07 | Médio | Tools validadas; sem SQL direto | ✓ Implementado |
| 8 | Model Theft | LLM08 | Médio | INT4 quantization (ofuscação) | ✓ Implementado |
| 9 | Unauthorized Code Execution | LLM09 | Crítico | Sandbox tools; sem `eval()` | ✓ Implementado |
| 10 | Model Poisoning (Fine-Tuning) | LLM10 | Médio | Fine-tuning desabilitado em produção | ✓ Implementado |


## 5. Logging Estruturado (sem PII)

### ✓ Bom

```python
import logging
import json

logger = logging.getLogger(__name__)

logger.info(json.dumps({
    "event": "agent_query",
    "user": user_hash,
    "tool_used": "stock_lookup",
    "latency_ms": 234,
    "status": "success",
    "timestamp": datetime.now().isoformat()
}))
```


## 6. Conformidade e Auditorias

### Logs Auditáveis

✓ Todas as predições são logadas com:
- User ID (anonimizado)
- Query (sanitizada)
- Resposta (sanitizada)
- Latência
- Timestamp
- IP origem


**Última atualização:** Abril 2026  
**Próxima revisão:** Junho 2026
