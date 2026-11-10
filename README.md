# PLO RangeLab

[![CI](https://github.com/bpodol/plorangelab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bpodol/plorangelab/actions/workflows/ci.yml)

**A poker range analysis and equity calculation application for Pot-Limit Omaha (PLO4 and PLO5).**

PLO RangeLab is a full-stack application built around a custom Python equity engine. It supports weighted hand ranges, multiway equity calculations, and range filtering.

**Status:** Under active development. The equity engine and core backend are implemented, while the frontend, filtering API, and persistence are being developed.

## Features

- **PLO4 and PLO5** support for 2–6 players.
- **Weighted hand ranges** with card-blocker handling.
- **Exact equity calculations** and adaptive Monte Carlo simulations.
- **Preflop, flop, turn, and river** calculations.
- **Range management and filtering** with grouped filters.
- **WebSocket-based equity calculations** with progress updates and cancellation.
- **Session-based REST API** for managing player ranges and boards.

## Tech Stack

**Backend:** Python, FastAPI, NumPy, Numba, Pydantic, WebSockets.

**Frontend:** React, TypeScript, Vite.

**Testing & Tools:** pytest, Ruff, uv, Git.

## Architecture

The application separates its components into:

- **Equity Engine:** Numba-compiled hand evaluation, exact enumeration, and Monte Carlo simulation.
- **Services:** Session management, player ranges, filters, and equity job execution.
- **API:** FastAPI REST endpoints and WebSocket communication.
- **Frontend:** React interface for interacting with the backend.

Equity calculations execute in dedicated worker threads to avoid blocking the API.

## Getting Started

Install Python dependencies:

```bash
uv sync
```

Start the backend:

```bash
uv run uvicorn plorangelab.api.main:app --reload
```

## Tests

Run the test suite:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

## Roadmap

- Complete filtering API and frontend integration.
- Implement database persistence.
- Finalize the MVP interface.
- Add targeted performance benchmarks and deployment documentation.

---
