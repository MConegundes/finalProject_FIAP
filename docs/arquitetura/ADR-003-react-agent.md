# ADR-003: Agente ReAct com Tools para Análise Financeira

**Data:** Abril 2026  
**Status:** Aceito  
**Contexto:** Integração de reasoning + acting para consultas sobre ações

## Problema

Usuários precisam fazer perguntas complexas sobre ações (ex: "A Tesla vai melhorar no próximo mês?") que exigem múltiplos passos de reasoning e acesso a dados em tempo real.

## Decisão

Implementar **Agente ReAct (Reasoning + Acting)** usando LangChain com ≥3 tools customizadas que interagem com dados do domínio.

### Justificativa

1. **Multipaço:** ReAct permite que o LLM pense sobre qual ferramenta usar antes de chamar
2. **Transparência:** Saída detalhada mostra o raciocínio (Thought → Action → Observation → Answer)
3. **Modularidade:** Tools podem ser adicionadas/removidas sem retreinar o LLM
4. **FIAP:** Framework maduro, documentado, usado em produção

### Arquitetura

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  LLM (Qwen 2.5 INT4)            │
│  REACT_PROMPT                   │
└────────┬────────────────────────┘
         │
         ├─► Thought: Preciso de dados recentes
         │
         ▼
    Tools Available:
    ├─ stock_lookup (yfinance)
    ├─ technical_analysis (SMA, RSI, volatilidade)
    ├─ price_prediction (LSTM)
    └─ rag_knowledge (contexto histórico)
         │
         ├─► Action: stock_lookup(TSLA)
         │
         ▼
    ┌──────────────────────────┐
    │  Observation             │
    │  TSLA: $195.50, +2.3%    │
    └──────────────────────────┘
         │
         ├─► Action: technical_analysis(TSLA)
         │
         ▼
    ┌──────────────────────────┐
    │  RSI=65, SMA20>SMA50      │
    │  Signal: ALTA             │
    └──────────────────────────┘
         │
         ├─► Thought: Tenho dados; devo prever
         │
         ▼
    ┌──────────────────────────┐
    │  Final Answer            │
    │  "Tesla em alta..."      │
    └──────────────────────────┘
         │
         ▼
    Output Guardrails (PII removal)
         │
         ▼
    User Response
```

### Tools 

#### 1. **stock_lookup** — Dados recentes

```python
def _stock_lookup(ticker: StockSymbol) -> str:
    """Retorna: preço atual, variação dia, volume."""
    # Fonte: yfinance (últimos 5 dias)
    # Output: "TSLA: $195.50\n+2.3%\nVolume: 50M"
```

**Casos de uso:**
- "Qual o preço da Tesla agora?"
- "Como está a BYD hoje?"

---

#### 2. **technical_analysis** — Indicadores técnicos

```python
def _technical_analysis(ticker: StockSymbol) -> str:
    """Retorna: SMA20, SMA50, RSI(14), volatilidade anualizada, sinal."""
    # Fonte: yfinance (últimos 3 meses)
    # Métricas: SMA, RSI, volatilidade
```

**Casos de uso:**
- "A Tesla está em tendência de alta?"
- "Qual o RSI da Toyota?"

---

#### 3. **price_prediction** — LSTM forecast

```python
def _price_prediction(ticker: StockSymbol) -> str:
    """Retorna: predições para próximos 5 dias."""
    # Fonte: Modelo LSTM treinado
    # Output: "Predição 5 dias: $200, $205, $210..."
```

**Casos de uso:**
- "Qual a tendência de preço da ação?"
- "A TSLA vai subir?"

---

#### 4. **rag_knowledge** — Contexto histórico (bonus)

```python
def _rag_knowledge(query: str) -> str:
    """Busca documentos contextuais (notícias, relatórios)."""
    # Fonte: ChromaDB com embeddings
    # Output: Contexto relevante em português
```

**Casos de uso:**
- "Qual foi o impacto da queda do mercado em 2023?"
- "Quais são os riscos do setor?"

### Prompt ReAct Customizado

```python
REACT_PROMPT = PromptTemplate.from_template("""
Você é um analista financeiro especializado em previsão de preços de ações.
Use as ferramentas disponíveis para responder perguntas sobre ações,
previsões e análises técnicas.

Ferramentas disponíveis:
{tools}

Use o formato:
Thought: pensar sobre o que fazer
Action: nome_da_ferramenta
Action Input: input para a ferramenta
Observation: resultado da ferramenta
... (repita Thought/Action/Observation quantas vezes necessário)
Thought: Agora sei a resposta final
Final Answer: resposta para o usuário

Pergunta: {input}
{agent_scratchpad}
""")
```

### Parâmetros do AgentExecutor

```python
AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,              # Log detalhado (desativar em produção)
    max_iterations=10,         # Evita loops infinitos
    handle_parsing_errors=True # Robustez contra alucinações do LLM
)
```

### Validação de Tools

Cada tool deve:

1. ✓ Ter `name`, `func`, `description` bem definidos
2. ✓ Tipo de entrada clara (StockSymbol ou str)
3. ✓ Tipo de saída sempre `str` (para LLM processar)
4. ✓ Error handling gracioso (retorna mensagem vs. exception)
5. ✓ Timeout (máx 10s por call)

## Consequências

### Positivas
✅ Transparência: usuário vê raciocínio do agente  
✅ Acesso a dados reais via tools  
✅ Escalável (adicionar tools é trivial)  
✅ Testável (mock tools para testes)  

### Negativas
❌ Latência: múltiplas calls a tools (2-5s por query)  
❌ Consumo de tokens: prompt longo (~500 tokens)  
❌ Dependências: yfinance, LSTM model, ChromaDB  

## Mitigação

- ✓ Cache de resultados (Redis) para queries repetidas
- ✓ Rate limiting em tools externas
- ✓ Fallback para baseline responses se erro crítico

## Monitoramento

**Métricas:**
- `agent_latency_p95` (target: <5s)
- `tool_success_rate` (target: >95%)
- `reasoning_steps_avg` (expectativa: 3-4 por query)
