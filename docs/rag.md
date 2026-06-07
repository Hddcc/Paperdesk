# PaperDesk RAG

PaperDesk RAG is scoped to paper-reading workflows. It should stay focused on evidence quality, selected document scope, and citation-aware answers.

Agent-facing ownership is documented through `backend/app/agent/lifecycle` and `backend/app/agent/runtimes`. The current paper-domain implementation remains in `backend/app/services/rag_service.py`, `backend/app/services/text_chunker.py`, `backend/app/services/embedding_service.py`, and repository/vector-store helpers.

## Ingestion

The paper library flow preserves:

- PDF upload,
- PDF text parsing,
- chunk generation,
- embedding generation,
- vector index writes,
- library metadata,
- category/tag relations.

Session file attachments are separate read-only context. They can support the current chat turn without becoming library documents.

## Retrieval

Paper RAG keeps these retrieval capabilities:

- vector recall,
- keyword fallback,
- metadata filtering,
- selected document range,
- evidence deduplication,
- citation metadata,
- evidence quality warnings.

The selected document ids in `ContextPacket.selected_document_ids` define the strongest paper scope. Retrieved snippets are carried through `ContextPacket.evidence`, finalized by runtime response metadata, and summarized in lifecycle trace events.

## Evidence Assembly

Evidence should preserve:

- document id,
- title or display name,
- page or section when available,
- citation label,
- snippet text,
- quality or warning metadata.

When support is insufficient, PaperDesk should state the evidence boundary in the answer rather than filling gaps with unsupported claims.

## Simplification Boundary

Query expansion, complex reranking, caching, cross-encoder rerankers, external web RAG, and multi-hop workflows should be enabled only when they improve a concrete workflow such as selected-paper review or research report generation.

Default chat should remain lightweight:

- direct chat uses no paper retrieval,
- paper RAG uses scoped retrieval plus synthesis,
- library reads use deterministic tools where possible,
- writes go through preview and confirmation.
