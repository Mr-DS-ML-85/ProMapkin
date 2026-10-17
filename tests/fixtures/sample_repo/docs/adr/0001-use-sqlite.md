# ADR 0001: Use SQLite for the document index

## Status

Accepted

## Context

The index must run locally without extra infrastructure.

## Decision

Store documents in SQLite through `IndexRepository`.

## Consequences

Concurrent writers are limited; `IndexRepository` serializes writes.
