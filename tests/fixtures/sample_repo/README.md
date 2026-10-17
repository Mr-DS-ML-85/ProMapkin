# Search App

A small search service. `SearchService` depends on `IndexRepository` and a `Ranker` strategy.

## Architecture

The HTTP API (`searchapp/api/routes.py`) calls `SearchService`, which reads documents from
`IndexRepository` and orders them with a `Ranker`. See [the architecture notes](docs/architecture.md).

## Setup

Install the requirements and set `API_TOKEN`.
