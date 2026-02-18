# Research Agent Roadmap (Content-First)

## Project Goal
- Build a vertical research agent that maximizes **evidence quality** before generation quality.
- Enforce a two-lane output model:
  - **Professional evidence lane**: papers, benchmarks, code, model/dataset cards.
  - **Community signal lane**: engineering discussion, adoption friction, risk sentiment.

## Retrieval Principles
- Query planning must produce dimension-complete coverage: architecture, performance, comparison, limitation, deployment.
- Prefer primary technical sources first, then community corroboration.
- Keep source failures non-blocking: one source down should not stop report/video generation.

## Mainstream Retrieval Patterns (to continue implementing)
- Hybrid retrieval and reciprocal rank fusion (BM25 + dense).
  - Ref: Azure AI Search hybrid retrieval: https://learn.microsoft.com/azure/search/hybrid-search-overview
- Multi-query expansion / decomposition.
  - Ref: LangChain multi-query retriever: https://python.langchain.com/docs/how_to/MultiQueryRetriever/
- Semantic + lexical reranking.
  - Ref: Elasticsearch RRF retriever: https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
- Scholarly API-first indexing (OpenAlex/Semantic Scholar/arXiv) and metadata-driven joins.
  - Ref: OpenAlex docs: https://docs.openalex.org/
  - Ref: Semantic Scholar API docs: https://api.semanticscholar.org/api-docs/graph
- Code evidence enrichment from repository artifacts (README/config/core modules).
  - Ref: GitHub REST search API: https://docs.github.com/rest/search/search

## Implemented In This Iteration
- Added user-supplied document ingestion:
  - arXiv URL -> metadata + optional PDF excerpt -> seed evidence.
  - PDF upload -> parsed excerpt -> seed evidence.
- Seed evidence is merged into the search pipeline before analysis/generation.
- Frontend now separates professional/community source summaries for transparency.
- Video formula page removed for stability; narration retime strategy changed to avoid abrupt slow speech.

## Near-Term Next Steps
- Add OpenAlex as additional paper recall channel and merge by DOI/arXiv ID.
- Add lightweight reranker after deep enrichment.
- Add explicit “evidence sufficiency gate” before video generation:
  - if low coverage -> force report-only mode with remediation tasks.
- Add quality telemetry dashboard:
  - source coverage, contradiction rate, quantitative-evidence ratio, replayable provenance.
