# Architecture

## Components

- `SearchService` uses `IndexRepository` for persistence.
- `VectorRanker` calls `EmbeddingAdapter` to embed text.
- `SearchService` writes to `EmbeddingAdapter`.

## Data flow

Requests arrive at `search_endpoint`, flow through `SearchService.search`, then `select_candidates`
and the configured `Ranker`.
