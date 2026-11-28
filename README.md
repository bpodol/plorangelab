# PLO RangeLab

[![CI](https://github.com/bpodol/plorangelab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bpodol/plorangelab/actions/workflows/ci.yml)

**A poker range analysis and equity calculation application for Pot-Limit Omaha (PLO4 and PLO5).**

PLO RangeLab is a full-stack application built around a custom Python equity engine. It supports weighted hand ranges, multiway equity calculations, and range filtering.

**Status:** Under active development. The equity engine, backend services, REST API, equity calculation UI, filter-management UI, database infrastructure, and backend user registration are implemented. User authentication, saved range persistence, and further UI improvements are in progress.

## Features

- **PLO4 and PLO5** support for two to six players, with a shared game mode per session.
- **Weighted hand ranges** with card-blocker handling.
- **Exact equity calculations** and adaptive Monte Carlo simulations.
- **Preflop, flop, turn, and river** calculations.
- **Interactive range filtering** with nested filter groups, individual activation controls, weighted coverage statistics, and filter management through the frontend.
- **Board-dependent filter generation** for newly created flop, turn, and river states.
- **HTTP + Server-Sent Events (SSE)** for equity-job submission, streamed progress/results, and cancellation.
- **Session-based REST API** for managing player ranges and boards, returning current-street filter snapshots when ranges are added or replaced.
- **User registration** with lowercase username normalization, Argon2id password hashing, and duplicate-username detection.

## Tech Stack

**Backend:** Python, FastAPI, NumPy, Numba, Pydantic, argon2-cffi, SSE.

**Database:** PostgreSQL, SQLAlchemy (async), Psycopg, Alembic.

**Frontend:** React, TypeScript, Vite.

**Infrastructure:** Docker, Docker Compose, GitHub Actions.

**Testing & Tools:** pytest, pytest-asyncio, Testcontainers, Vitest, Ruff, Oxlint, Prettier, uv, Git.

## Architecture

The application separates its components into:

- **Equity Engine:** Numba-compiled hand evaluation, exact enumeration, and Monte Carlo simulation.
- **Services:** User registration, session management, shared street states, automatic filter initialization, player-specific range filtering, and equity job execution.
- **API:** FastAPI HTTP endpoints for job submission/cancellation and SSE for equity updates, alongside REST routes for user registration and session/range/filter management.
- **Database Infrastructure:** PostgreSQL schema, asynchronous SQLAlchemy engine and session factories, and Alembic-managed migrations.
- **Frontend:** React interface for interacting with the backend.

Equity calculations execute in dedicated worker threads to avoid blocking the API.

FastAPI initializes the asynchronous database engine and session factory at startup. User registration persists accounts to PostgreSQL; connections are opened lazily when needed. Application analysis sessions, player ranges, filter states, and equity jobs remain in memory.

## Documentation

Technical documentation is organized by subsystem:

- [Backend Architecture](docs/backend/README.md) — Backend components,
  responsibilities, and execution flows.
- [Database Infrastructure](docs/backend/database.md) — Database schema, asynchronous connectivity, migrations, development setup, and integration testing.
- [Frontend Architecture](docs/frontend/README.md) — React application
  structure, component responsibilities, and workflows.
- [Frontend State and Data Flow](docs/frontend/state-and-data-flow.md) —
  State ownership, HTTP job requests, and SSE lifecycle.
- [REST API](docs/api/rest.md) — Endpoints, request and response schemas,
  and error handling.
- [Equity SSE Protocol](docs/api/sse.md) — Equity job submission,
  streamed status/progress/results, cancellation, and reconnection.


## Getting Started

### Requirements

- Python and [uv](https://docs.astral.sh/uv/)
- [Node.js 24 LTS](https://nodejs.org/) and npm
- [Docker](https://docs.docker.com/get-docker/) with Docker Compose support (required for local PostgreSQL and database integration tests).

### Backend

From the repository root, install dependencies:

```bash
uv sync
```

Configure the database before running the backend locally. Create a `.env` file in the repository root using `.env.example` as a template and configure the database credentials and `DATABASE_URL`.

Start PostgreSQL:

```bash
docker compose up -d
```

Apply database migrations:

```bash
uv run --env-file .env alembic upgrade head
```

Start the FastAPI development server:

```bash
uv run --env-file .env uvicorn plorangelab.api.main:app --reload
```

`DATABASE_URL` is required at startup. Engine initialization does not check database connectivity; registration requires a reachable, migrated PostgreSQL database. If the environment is already configured, `--env-file .env` is unnecessary.

For detailed configuration, migration commands, and storage behavior, see [Database Infrastructure](docs/backend/database.md).

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

Run the complete backend test suite:

```bash
uv run pytest
```

Database integration tests require a running Docker daemon. Testcontainers automatically provisions a disposable PostgreSQL instance, applies Alembic migrations, and removes the container after the test session.

Run database integration tests separately:

```bash
uv run pytest tests/integration/
```

No manually configured test database or test environment file is required.

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

- Implement user login, logout, and authentication.
- Associate application sessions with authenticated users.
- Implement range serialization and saved range persistence.
- Add straight draw automatic filter generation.
- Finalize the MVP interface.
- Add targeted performance benchmarks and deployment documentation.

---
