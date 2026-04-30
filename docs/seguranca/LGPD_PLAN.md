# LGPD PLAN — Conformidade com Lei Geral de Proteção de Dados

## 1. Visão Geral

Sistema de **previsão de preços de ações** com integração de **Agente LLM** processa dados pessoais (queries de usuários). Este plano detalha conformidade com LGPD.

### Dados Pessoais Processados

| Tipo | Descrição | Base Legal |
|------|-----------|-----------|
| **User ID** (anon) | Hash SHA256 do usuário | Legítimo interesse |
| **Queries** | Perguntas sobre ações | Consentimento (termos de uso) |
| **IP Origin** | Endereço IP de requisição | Legítimo interesse (segurança) |
| **Timestamps** | Quando a query foi feita | Legítimo interesse |

---

## 2. Base Legal

### Art. 7 — Fundamentos de Licitude

Qual base legal justifica o processamento?

| Dado | Base | Justificativa |
|------|------|---------------|
| User ID (hash) | VI - Legítimo interesse | Análise de uso e billing |
| Queries | II - Consentimento | Termos de uso aceitam coleta anônima |
| IP/Timestamp | VI - Legítimo interesse | Segurança, detecção de fraude |
| Predições de preço | I - Consentimento | Serviço solicitado |

### Consentimento

```
TERMO DE CONSENTIMENTO PARA PROCESSAMENTO DE DADOS

Você concorda que suas queries ao agente de análise de ações serão:

1. Coletadas em forma anônima (apenas hash do ID)
2. Armazenadas por até 30 dias para melhoria de serviço
3. Analisadas para detectar padrões de uso (sem profiling individual)
4. Deletadas automaticamente após 30 dias

Você pode revogar este consentimento a qualquer momento via:
   - POST /revoke_consent
   - Email: dpo@datathon.fiap.com.br

☐ Aceito os termos acima
```

## 3. Responsabilidades

### 3.1. Controlador (Datathon / FIAP)

Responsável por:
- ✓ Fundamentar licitude do processamento
- ✓ Implementar direitos dos titulares
- ✓ Manter registros de processamento
- ✓ Designar DPO

**Ações:**
```
1. Publicar aviso de privacidade (website)
2. Implementar endpoints de exercício de direitos
3. Manter log de consentimentos
4. Auditar anualmente
```

### 3.2. DPO (Data Protection Officer)

```

Responsabilidades:
✓ Fiscalizar conformidade
✓ Atender requisições de direitos
✓ Comunicar com autoridades (ANPD)
```

---

## 4. Segurança dos Dados

### Art. 32 — Medidas de Segurança

| Medida | Implementação | Status |
|--------|---------------|--------|
| **Encriptação em trânsito** | TLS 1.3 (HTTPS) | ✓ Implementada |
| **Encriptação em repouso** | AES-256 (dados pessoais) | ✓ Implementada |
| **Controle de acesso** | JWT + RBAC | ✓ Implementada |
| **Autenticação multi-fator** | 2FA para admin | ⏳ Roadmap |
| **Pseudonimização** | Hash SHA256 user IDs | ✓ Implementada |
| **Backup** | Daily replication em S3 | ✓ Implementada |
| **Testes de segurança** | Penetration testing | ⏳ Q2 2026 |

---

## 5. Incidentes e Notificações (Art. 34)

### Plano de Resposta a Incidentes

```
VAZAMENTO DE DADOS PESSOAIS
├─ Detectado
├─ Isolar sistema (desligar API)
├─ Investigar (DPO + Security)
├─ Mitigação (patch)
├─ Notificar ANPD (48h) se risco alto
├─ Notificar titulares (30 dias)
└─ Relatório post-mortem
```


## 6. Avaliação de Impacto (AIPD) — Art. 32

```
QUESTÕES CRÍTICAS:

1. Pode causar dano aos titulares?
   Resposta: Não. Previsões financeiras são informativas, não vinculantes.
   
2. Há processamento em larga escala?
   Resposta: Não. <100k usuários ativos.
   
3. Há profiling ou scoring?
   Resposta: Não. Apenas análise técnica (sem decisões sobre pessoa).
   
4. Tecnologia inovadora?
   Resposta: Sim (LLM). Monitoramento necessário.
   
CONCLUSÃO: AIPD completa recomendada em Q3 2026.
```

---

## 7. Checklist de Conformidade

### Pré-Launch

- [ ] Publicar Política de Privacidade (website)
- [ ] Implementar endpoints de direitos (acesso, deleção)
- [ ] Designar DPO formalmente
- [ ] Treinar equipe em LGPD
- [ ] Documentar base legal para cada dado
- [ ] Configurar retenção automática (30 dias)
- [ ] TLS 1.3 obrigatório em APIs
- [ ] Audit log ativado

### Lançamento

- [ ] Notificar usuários (aviso de privacidade)
- [ ] Obter consentimento explícito
- [ ] Testes de exercício de direitos
- [ ] Monitoramento 24/7

### Pós-Launch (Mensal)

- [ ] Relatório de requisições de direitos
- [ ] Audit de retenção de dados
- [ ] Teste de backup/recovery
- [ ] Scan de vulnerabilidades

### Anual

- [ ] AIPD completa
- [ ] Avaliação de conformidade externa
- [ ] Atualização de política de privacidade
- [ ] Rotação de secrets/keys

---

## Referências Legais

1. Lei nº 13.709/2018 — Lei Geral de Proteção de Dados (LGPD)
2. Resolução ANPD nº 1/2021 — Diretrizes
3. Guia de Impacto da ANPD (2021)
4. GDPR Art. 35 (modelo de AIPD)

---

**Aprovado por:** [DPO]  
**Última atualização:** Abril 2026  
**Próxima revisão:** Janeiro 2027
