# Feedback & Changelog — v1.0 → v2.0

This document captures the architectural gap analysis that drove the v2.0 upgrade, and maps each identified weakness to its resolution.

## Gap → Resolution Map

### 1. Layout-Blind Chunking → Layout-Aware Semantic Parser

**Gap:** Raw text extraction flattened tables, split captions from their figures, and allowed multi-page tables to be fragmented into meaningless chunks.

**Resolution (`src/rag_system/components/layout_parser.py`):**
- `LayoutAwareParser` groups elements by semantic proximity before chunking
- Tables are matched with their captions within a configurable line-proximity window
- Consecutive table elements on adjacent pages are merged into single `LayoutChunk` objects with `is_continuation=True`
- Section headings are detected (Item 7, RISK FACTORS, NOTE 12, etc.) and propagated to all child chunks as `heading` metadata
- All structured elements are wrapped in HTML (`<table>`, `<figure>`, `<section>`) to give the LLM rich layout context
- `to_document_elements()` converts back to `DocumentElement` with HTML embedded in the `text` field

---

### 2. Hallucinated Arithmetic → Program-of-Thought Sandboxed Executor

**Gap:** LLMs computing multi-step financial formulas in natural language introduce errors. Direct arithmetic in generation is unreliable.

**Resolution (`src/rag_system/components/pot_executor.py`):**
- `ASTSandboxValidator` validates generated code against an AST node allowlist before execution — blocks `import`, `exec`, `eval`, `open`, `__builtins__` access, and all network libraries
- `PoTExecutor.execute_code()` runs validated code in a restricted namespace with only safe builtins (`round`, `abs`, `sum`, `min`, `max`, `float`, `int`, etc.)
- `asyncio.wait_for()` enforces a hard 5-second timeout preventing infinite loop DoS
- Named financial templates (`cagr`, `roi`, `gross_margin`, `percentage_change`, `ebitda_margin`, `debt_to_equity`) are provided for direct use without LLM code generation
- Code extraction supports fenced (` ```python `) and unfenced formats

---

### 3. Hardcoded Components → Pluggable ABC Architecture

**Gap:** Components (parser, vector store, LLM) were tightly coupled to specific implementations.

**Resolution (`src/rag_system/components/base.py`):**
- `BaseParser`, `BaseEmbedder`, `BaseVectorStore`, `BaseRetriever`, `BaseReranker`, `BaseGenerator`, `BaseEvaluator` define strict contracts via `@abc.abstractmethod`
- Shared data models (`DocumentElement`, `RetrievedChunk`, `GeneratedAnswer`) are immutable Pydantic v2 models
- All components accept base types — the pipeline orchestrator never imports concrete implementations
- Factory functions (`build_parser()`, `build_vector_store()`, `build_reranker()`, etc.) map config values to concrete classes
- `InMemoryVectorStore` provides a zero-dependency alternative for testing and development

---

### 4. Single-Provider Retrieval → Hybrid RRF + Reranking

**Gap:** Dense-only retrieval missed exact financial figure matches (specific numbers, tickers, dates).

**Resolution (`src/rag_system/components/retriever/__init__.py`):**
- `HybridRetriever` combines dense vector search + in-memory `BM25Index`
- Fusion via `_reciprocal_rank_fusion()` — no weight tuning required, robust to score scale differences
- `CrossEncoderReranker` (local, `ms-marco-MiniLM-L-6-v2`) and `CohereReranker` (cloud) provide final relevance scoring
- Configurable via `RETRIEVER_CONFIG__STRATEGY=hybrid` and `RERANKER_CONFIG__PROVIDER=cross_encoder`

---

### 5. No Evaluation Gate → RAGAS + CI Regression Blocker

**Gap:** No automated quality measurement prevented regressions from reaching production.

**Resolution (`src/rag_system/components/evaluator/__init__.py`, `.github/workflows/ci.yml`):**
- `RagasEvaluator` wraps RAGAS (faithfulness, answer_relevancy, context_precision) with LLM-as-judge numeric accuracy scoring
- `GoldenDatasetRunner` loads JSONL golden samples, runs pipeline, scores each, and produces `EvalReport`
- History tracking in `evals/history.json` enables regression detection (>5% faithfulness drop fails CI)
- GitHub Actions workflow runs eval gate on every PR with `--fail-on-regression`

---

### 6. No Observability → OTel + Prometheus + Grafana

**Gap:** No distributed tracing, no custom RAG metrics, no cost visibility.

**Resolution (`src/rag_system/utils/telemetry.py`, `grafana/dashboards/`):**
- OpenTelemetry spans: `ingest_pipeline`, `parse_document`, `vision_describe`, `retrieval`, `llm_generation`
- Prometheus custom metrics: `rag_query_latency_seconds`, `rag_hallucination_score`, `rag_citation_coverage_ratio`, `rag_query_cost_usd_total`, `rag_tokens_total`, `rag_cache_hits_total`
- Grafana dashboard JSON auto-provisioned with 10 panels covering latency percentiles, cost, token consumption, hallucination score, cache hit rate

---

### 7. No Security Layer → Auth + PII + Guardrails + Audit

**Gap:** No authentication, no PII protection on ingested content, no tamper-evident audit trail.

**Resolution:**
- `APIKeyMiddleware` — constant-time SHA-256 key comparison, exempt paths for health probes
- `PIIRedactor` — Presidio integration with financial-domain extensions (CUSIP, ISIN, routing numbers)
- `FinancialGuardrails` — numeric grounding check (every answer number must appear in context), prompt injection detection
- `AuditLogger` — append-only JSONL with SHA-256 content hash per event, GDPR/CCPA deletion logging

---

### 8. No Multi-Tenancy → Per-Tenant Isolation + Quotas + Cost Tracking

**Gap:** Single shared vector store with no tenant separation.

**Resolution:**
- Per-tenant dataset paths in DeepLake (`rag_financial_{tenant_id}`)
- `AsyncRateLimiter` maintains per-tenant token buckets with global cap
- `CostTracker` accumulates per-tenant, per-model USD costs with monthly quota enforcement
- All log records, audit events, and Prometheus labels carry `tenant_id`

---

### 9. Notebook-Only Interface → CLI + REST API + SDK

**Gap:** System only usable via Jupyter notebook — no programmatic or operational interface.

**Resolution:**
- `rag-financial` CLI (`typer`) — `ingest`, `query`, `evaluate`, `serve`, `health` commands with `rich` output
- FastAPI application with `/api/v1/ingest`, `/api/v1/query`, `/api/v1/tenants`, `/healthz`, `/readyz` endpoints
- Python SDK (`src/rag_system/sdk`) with both `async` and sync wrappers, YAML config loader

---

### 10. No Production Infrastructure → Docker + K8s + Helm

**Gap:** No deployment artifacts — system only ran locally.

**Resolution:**
- Multi-stage Dockerfile (builder → non-root runtime, HEALTHCHECK, Prometheus port)
- `docker-compose.yml` with Redis, OTel collector, Prometheus, Grafana, Jaeger
- Kubernetes base manifests: Deployment, Service, HPA (2–10 replicas), ConfigMap, PVC, ServiceMonitor
- Helm chart with `values.yaml`, HPA, PodDisruptionBudget, Prometheus ServiceMonitor
- GitHub Actions CI: lint → test → security scan → eval gate → Docker build → Helm lint
