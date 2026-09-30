# ADR-0005: AI Security Guardrails, WAF & Data Loss Prevention (DLP)

## Status
**Accepted**

## Context
Deploying Generative AI and Retrieval-Augmented Generation (RAG) in corporate environments introduces critical cybersecurity attack surfaces documented in the **OWASP Top 10 for LLM Applications** and **OWASP API Security Top 10**:
1. **Prompt Injection & Jailbreaks (LLM01)**: Malicious queries attempting to override system instructions, extract proprietary system prompts, or manipulate context retrieval.
2. **Sensitive Information Disclosure (LLM06)**: Accidental leakage or indexing of Personally Identifiable Information (PII), payment card numbers (PAN), Social Security Numbers (SSN), and API keys.
3. **API Exploits & Resource Exhaustion (API4, API8)**: Path traversal, XSS, malicious scanner probes, and unconstrained request body sizes leading to memory starvation.

## Decision
We implement a multi-layered **Defense-in-Depth AI Cybersecurity Architecture**:

### 1. Perimeter Edge Protection (Go API Gateway)
- **Hardened HTTP Response Headers**: Enforces Strict-Transport-Security (HSTS), Content-Security-Policy (CSP), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and `Referrer-Policy`.
- **Lightweight WAF Middleware**: Inspects paths and query parameters for Path Traversal (`../`), Command Injection, XSS, SQLi signatures, and automated scanners (e.g. `sqlmap`, `nikto`).
- **Anti-DoS Payload Limiting**: Restricts maximum request body size (5MB).

### 2. AI Semantic Guardrails (Hybrid RAG Engine)
- **Prompt Injection Detector**: Pre-retrieval scanner inspecting queries against direct instruction overrides, delimiter hijacking (`[INST]`, `### System:`), roleplay jailbreaks (DAN), and Base64-obfuscated injection vectors.
- **Threat Scoring**: Computes a confidence threat score; queries exceeding the threshold are blocked immediately with `HTTP 400 Bad Request` and an explicit security violation code.

### 3. Data Loss Prevention & PII Masking (DLP)
- **Dual-Phase Sanitization**:
  - *Ingestion Phase*: Strips and redacts sensitive PII before text is vectorized or persisted in Qdrant.
  - *Egress Phase*: Post-synthesis sanitizer guarantees no synthesized LLM answers leak unmasked secrets.
- **Luhn Algorithm Verification**: Validates potential 13-19 digit card numbers to avoid false positives and mask real PANs with `[REDACTED_CREDIT_CARD]`.
- **Token Redaction**: Detects SSNs, API tokens (`sk-*`, `ghp_*`, `Bearer`), corporate emails, and phone numbers.

### 4. SIEM & Audit Telemetry
- All security blocks emit structured `[SECURITY_AUDIT]` log lines containing `client_ip`, `trace_id`, violation rules, and threat scores for seamless ingestion into enterprise SIEMs (Splunk, Datadog, Elastic).
- Prometheus security counters (`rag_security_events_total`) expose real-time metrics for SOC alerting.

## Consequences
### Positive
- Prevents proprietary knowledge leakage and jailbreak manipulation.
- Ensures compliance with privacy standards (GDPR, PCI-DSS, SOX, HIPAA).
- Negligible computational overhead (<1.5ms per request).

### Negative / Trade-offs
- Security regex rules require regular maintenance as novel jailbreak taxonomies emerge.
