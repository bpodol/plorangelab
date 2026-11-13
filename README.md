# PLO RangeLab

[![CI](https://github.com/bpodol/plorangelab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bpodol/plorangelab/actions/workflows/ci.yml)

**A poker range analysis and equity calculation application for Pot-Limit Omaha (PLO4 and PLO5).**

PLO RangeLab is a full-stack application built around a custom Python equity engine. It supports weighted hand ranges, multiway equity calculations, and range filtering.

**Status:** Under active development. The equity engine, backend services, REST API, and core equity calculation UI are implemented. Frontend filter management and database persistence are in progress.

## Features

- **PLO4 and PLO5** support for 2–6 players.
- **Weighted hand ranges** with card-blocker handling.
- **Exact equity calculations** and adaptive Monte Carlo simulations.
- **Preflop, flop, turn, and river** calculations.
- **Range management and backend filtering** with nested filter groups.
- **WebSocket-based equity calculations** with progress updates and cancellation.
- **Session-based REST API** for managing player ranges and boards.

## Tech Stack

**Backend:** Python, FastAPI, NumPy, Numba, Pydantic, WebSockets.

**Frontend:** React, TypeScript, Vite.

**Testing & Tools:** pytest, Vitest, Ruff, Oxlint, Prettier, uv, Git.

## Architecture

The application separates its components into:

- **Equity Engine:** Numba-compiled hand evaluation, exact enumeration, and Monte Carlo simulation.
- **Services:** Session management, player ranges, filters, and equity job execution.
- **API:** FastAPI REST endpoints and WebSocket communication.
- **Frontend:** React interface for interacting with the backend.

Equity calculations execute in dedicated worker threads to avoid blocking the API.

## Documentation

Technical documentation is organized by subsystem:

- [Backend Architecture](docs/backend/README.md) — Backend components,
  responsibilities, and execution flows.
- [Frontend Architecture](docs/frontend/README.md) — React application
  structure, component responsibilities, and workflows.
- [Frontend State and Data Flow](docs/frontend/state-and-data-flow.md) —
  State ownership, API communication, and WebSocket lifecycle.
- [REST API](docs/api/rest.md) — Endpoints, request and response schemas,
  and error handling.
- [WebSocket Protocol](docs/api/websocket.md) — Equity calculation
  commands, progress updates, cancellation, and reconnection.


## Getting Started

### Requirements

- Python and [uv](https://docs.astral.sh/uv/)
- [Node.js 24 LTS](https://nodejs.org/) and npm

### Backend

From the repository root, install dependencies:

```bash
uv sync
```

Start the FastAPI development server:

```bash
uv run uvicorn plorangelab.api.main:app --reload
```

### Frontend

In a separate terminal, from the repository root:

```bash
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite.


## Tests and Code Quality

### Backend

Run the backend tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Check formatting:

```bash
uv run ruff format --check .
```

### Frontend

From the `frontend/` directory:

Run the frontend tests:

```bash
npm test
```

Run linting:

```bash
npm run lint
```

Check formatting:

```bash
npm run format:check
```

Run the TypeScript compilation and production build:

```bash
npm run build
```

## Roadmap

- Implement frontend filter management.
- Generate board-dependent filter groups automatically.
- Implement database persistence.
- Finalize the MVP interface.
- Add targeted performance benchmarks and deployment documentation.

---
