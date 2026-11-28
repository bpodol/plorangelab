# Database infrastructure

## Overview

PLO RangeLab uses PostgreSQL for persistent user accounts and defines a schema for future saved poker ranges. The `db` package defines the database schema and provides factories for asynchronous database access. Alembic manages schema creation and evolution independently of the application runtime.

FastAPI owns the asynchronous engine and session factory. User registration uses request-scoped database sessions and persists user accounts through the user repository. Login, authentication sessions, saved-range operations, and range serialization are not implemented.

Application sessions, player range states, boards, filter trees, and equity jobs remain in memory. Their lifecycle is described in [Sessions and equity jobs](services/session-and-jobs.md) and [Range management](services/range-management.md).

## Components and responsibilities

| Component | Responsibility |
| --- | --- |
| PostgreSQL 18 | Persistent storage, SQL execution, transactions, and constraint enforcement |
| Docker Compose | Provision the local development database and its persistent storage |
| SQLAlchemy | ORM mapping, asynchronous database access, connection pooling, and transaction coordination |
| Psycopg 3 | PostgreSQL driver supporting synchronous and asynchronous connections |
| Alembic | Versioned database schema migrations through a synchronous SQLAlchemy engine |
| Testcontainers | Provision disposable PostgreSQL instances for integration tests |
| pytest-asyncio | Execute asynchronous integration tests and fixtures |

Database provisioning, schema migration, and runtime database access have independent lifecycles. The development database is managed by Docker Compose, whereas integration tests provision their own database through Testcontainers.

## Source files

| Location | Primary role |
| --- | --- |
| [`db/session.py`](../../src/plorangelab/db/session.py) | Asynchronous engine and session factories |
| [`db/models.py`](../../src/plorangelab/db/models.py) | Shared SQLAlchemy declarative base (`Base`), schema metadata, and `User` and `SavedRange` ORM models |
| [`db/repositories/users.py`](../../src/plorangelab/db/repositories/users.py) | User lookup and insertion operations |
| [`config.py`](../../src/plorangelab/config.py) | Application settings, including the required `DATABASE_URL` |
| [`compose.yaml`](../../compose.yaml) | Development PostgreSQL service and persistent volume |
| [`alembic.ini`](../../alembic.ini) | Alembic configuration |
| [`alembic/env.py`](../../alembic/env.py) | Migration environment, database URL, and target schema metadata |
| [`alembic/versions/`](../../alembic/versions/) | Versioned schema migration operations |
| [`tests/integration/conftest.py`](../../tests/integration/conftest.py) | PostgreSQL provisioning, migration setup, and test isolation fixtures |
| [`tests/integration/test_db.py`](../../tests/integration/test_db.py) | Database connectivity, persistence, and constraint tests |

Source paths for the `db` modules are relative to `src/plorangelab/`.

## Database schema

```text
users                               ranges
-----                               ------
id (PK)  ────── one-to-many ──────► user_id (FK)
username                            id (PK)
password_hash                       name
created_at                          mode
                                    range_blob
                                    created_at
                                    updated_at
```

### Users

ORM model: `User`. Table: `users`.

| Column | PostgreSQL type | Constraints and defaults |
| --- | --- | --- |
| `id` | `UUID` | Primary key; database-generated with `gen_random_uuid()` |
| `username` | `TEXT` | Not null; unique (`uq_users_username`) |
| `password_hash` | `TEXT` | Not null |
| `created_at` | `TIMESTAMP WITH TIME ZONE` | Not null; defaults to `now()` |

The table stores account identity and password hashes, not authentication sessions. Password hashing and verification are implemented in `security/passwords.py`. The database does not perform hashing or validate the hash format. Registration normalizes usernames to lowercase before insertion; the database does not independently enforce lowercase storage.

### Saved ranges

ORM model: `SavedRange`. Table: `ranges`.

| Column | PostgreSQL type | Constraints and defaults |
| --- | --- | --- |
| `id` | `UUID` | Primary key; database-generated with `gen_random_uuid()` |
| `user_id` | `UUID` | Not null; references `users.id` with `ON DELETE CASCADE` |
| `name` | `TEXT` | Not null |
| `mode` | `SMALLINT` | Not null; restricted to 4 or 5 |
| `range_blob` | `BYTEA` | Not null |
| `created_at` | `TIMESTAMP WITH TIME ZONE` | Not null; defaults to `now()` |
| `updated_at` | `TIMESTAMP WITH TIME ZONE` | Not null; defaults to `now()`; SQLAlchemy supplies `now()` on applicable updates |

Additional constraints:

- `ck_ranges_mode`: restricts saved ranges to PLO4 and PLO5.
- `uq_ranges_user_name_mode`: enforces uniqueness of `(user_id, name, mode)`. A user can reuse a range name in different modes, and different users can use identical names and modes.
- The foreign key prevents ranges from referencing nonexistent users. Deleting a user deletes all associated ranges at the database level, without requiring ORM relationships or individual range deletion statements.

PostgreSQL creates indexes for primary keys and unique constraints. The composite unique index begins with `user_id`, allowing it to support queries that filter by owner without an additional owner-only index at present.

The database stores range contents as opaque binary data. The serialization format and conversion between binary data and computational range representations such as `EncodedRange` are not yet defined. See [Card and range encoding](encoding.md) for the in-memory representations.

`created_at` and `updated_at` are generated by PostgreSQL on insertion when omitted. The `updated_at` update expression is configured in the ORM using `onupdate=func.now()`; it is not a database trigger and does not automatically apply to raw SQL updates.

### Integrity versus application validation

Database constraints protect structural integrity independently of the caller. They do not enforce application authorization, username normalization, nonempty text, valid password hashes, or the contents of a range blob. These checks belong to the application layer. In particular, an existing `user_id` is not proof that the current caller owns the referenced user or range.

## Engine, sessions, and transactions

`db/session.py` exposes two factory functions:

```python
create_db_engine(database_url: str) -> AsyncEngine
create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]
```

The engine factory uses `create_async_engine()` with `pool_pre_ping=True`. The session factory binds new sessions to the supplied engine and sets `expire_on_commit=False`.

```text
Database URL (supplied by caller)
└── create_db_engine()
    └── AsyncEngine
        ├── PostgreSQL dialect and asynchronous Psycopg driver
        ├── connection pool
        └── create_session_factory(engine)
            └── async_sessionmaker
                ├── AsyncSession (unit of work A)
                └── AsyncSession (unit of work B)
```

| Resource | Responsibility | Intended lifetime |
| --- | --- | --- |
| `AsyncEngine` | Database connectivity and connection pooling | Backend process |
| `async_sessionmaker` | Create consistently configured sessions | Backend process |
| Database connection | Execute SQL against PostgreSQL | Checked out when needed; returned to the pool afterward |
| `AsyncSession` | ORM object tracking, query execution, and transaction coordination | Independent unit of work |
| Transaction | Commit or roll back a group of database operations | Until commit or rollback |

The factories receive their dependencies explicitly. Importing the module does not require a database URL or initialize a global engine. This allows tests to supply a dynamically generated database URL without changing application configuration.

The engine opens connections lazily. `pool_pre_ping=True` checks connection liveness on checkout; it does not retry failed transactions. `expire_on_commit=False` keeps loaded ORM attributes accessible after a commit without automatically expiring them.

Example of using the factories directly (not a FastAPI request dependency):

```python
import os

from sqlalchemy import text

from plorangelab.db.session import create_db_engine, create_session_factory


async def check_database() -> None:
    engine = create_db_engine(os.environ["DATABASE_URL"])
    session_factory = create_session_factory(engine)

    try:
        async with session_factory() as session:
            result = await session.execute(text("SELECT current_database()"))
            print(result.scalar_one())
    finally:
        await engine.dispose()
```

The engine and session factory may be shared by tasks using the same application event loop. An `AsyncSession` must not be shared between concurrently executing tasks. The caller owns session lifetime and transaction boundaries; the factory does not automatically commit changes.

Closing a session releases its ORM and transactional resources, rolling back any remaining uncommitted transaction. Disposing of the engine releases idle pooled connections. FastAPI owns engine initialization and disposal through its lifespan. It loads `DATABASE_URL` from application settings, creates the engine and session factory, and exposes the factory through `app.state.db_session_factory`. Startup does not execute a database connectivity check.

A SQLAlchemy session is unrelated to an application analysis session or an authentication session.

### Request-scoped sessions and transaction ownership

- `api/dependencies.py`: `get_db_session()` creates an `AsyncSession` from the application session factory, yields it through `DbSession`, and closes it when the request dependency exits. It does not automatically commit.
- `services/users.py`: `register_user()` owns the transaction using `async with session.begin()`. It expects a session without an active transaction, hashes the password before starting the transaction, and returns only after successful transaction completion. Exceptions inside the transaction cause rollback; the service translates the username uniqueness violation after leaving the transaction context.
- `db/repositories/users.py`: repository functions execute queries and flush writes without committing or rolling back the caller's transaction.

### User repository

| Function | Contract |
| --- | --- |
| `get_user_by_id(session, user_id)` | Retrieve a user by UUID or return `None`. |
| `get_user_by_username(session, username)` | Retrieve a user by the supplied username or return `None`; does not normalize the input. |
| `create_user(session, username, password_hash)` | Add and flush a user, then return the ORM object; does not commit. |

Repositories accept already-prepared data and propagate database errors. For registration behavior, see the [services overview](services/README.md#user-registration).

## Local development setup

Run the following commands from the repository root. Docker with Compose support must be installed and the Docker daemon must be running.

Install backend dependencies:

```bash
uv sync --locked --dev
```

Create a local `.env` using `.env.example` as a template. The development configuration has the following variables:

```dotenv
POSTGRES_USER=dev_user
POSTGRES_PASSWORD=dev_password
POSTGRES_DB=plorangelab
DATABASE_URL=postgresql+psycopg://dev_user:dev_password@127.0.0.1:5433/plorangelab
```

Replace the example credentials with your local values. The credentials, database name, and host port in `DATABASE_URL` must match the Compose configuration. Percent-encode URL-reserved characters in connection URL credentials when necessary.

Docker Compose uses `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` to initialize PostgreSQL. Python and Alembic use `DATABASE_URL`. Compose reads `.env` for variable substitution; Python does not automatically load that file. Use `uv run --env-file .env` when a command requires the local database configuration.

Start the database:

```bash
docker compose up -d
```

The development service maps host port `5433` to PostgreSQL's container port `5432`. Using port `5433` avoids conflicts with a native PostgreSQL installation on the default port. For local development, keep the published port bound to `127.0.0.1`.

Check container status and database readiness:

```bash
docker compose ps
docker compose exec db pg_isready
```

Once PostgreSQL is ready, apply the migrations:

```bash
uv run --env-file .env alembic upgrade head
```

The Docker image initializes the database and its initial role; Alembic creates the application tables. Do not create application tables manually or use `Base.metadata.create_all()` as a substitute for migrations.

Start the API after applying migrations:

```bash
uv run --env-file .env uvicorn plorangelab.api.main:app --reload
```

### Persistent storage

The development database uses the named `postgres_data` volume mounted at `/var/lib/postgresql`, matching the PostgreSQL 18 image layout.

```bash
docker compose down
```

This removes the container but preserves the named volume. Starting the service again reuses the existing database files. The initialization variables are only applied when the data directory is empty; changing them does not automatically modify an existing database or role.

**Warning:** `docker compose down -v` also removes the named volume and destroys the development database contents. Only use it when intentionally resetting disposable local data.

The database role initialized by the Docker image is suitable for local development. Restricted runtime privileges, backups, and production database provisioning are outside the current scope.

## Schema migrations

`Base` is declared in `db/models.py`, alongside the ORM models. Its `Base.metadata` describes the current desired schema. Alembic migration files describe historical changes to that schema.

`alembic/env.py` imports the models so that their table definitions are registered in the shared metadata, sets `target_metadata` to `Base.metadata`, and obtains the database URL from the `DATABASE_URL` environment variable. It escapes percent characters when passing the URL through Alembic's configuration parser.

Alembic uses the standard synchronous migration environment. Psycopg supports both connection modes, so migrations and asynchronous application access can use the same `postgresql+psycopg://` URL while maintaining independent engines.

The initial migration, `3099a3c3a476`, creates `users` before `ranges` because the latter references the former. Downgrading removes the dependent table before the referenced table. Alembic maintains the `alembic_version` table to track the applied revision.

Revision `c362125c26e7` renames the username uniqueness constraint from `users_username_key` to `uq_users_username` without rebuilding the existing unique index. Its downgrade restores the original name. The explicit name is used by the registration service to distinguish username conflicts from unrelated integrity errors.

### Changing the schema

1. Modify the ORM model definitions.
2. Generate a candidate migration against the local development database:

   ```bash
   uv run --env-file .env alembic revision --autogenerate -m "describe schema change"
   ```

3. Review the generated `upgrade()` and `downgrade()` operations, including constraints, defaults, dependencies, and effects on existing data. Autogeneration does not reliably identify every schema change, such as table or column renames.
4. Apply the reviewed migration:

   ```bash
   uv run --env-file .env alembic upgrade head
   ```

5. Check the database revision and detectable differences from model metadata:

   ```bash
   uv run --env-file .env alembic current
   uv run --env-file .env alembic check
   ```

6. Run the integration tests and commit the model changes together with the migration.

`alembic check` detects outstanding changes supported by autogeneration; it is not a complete schema equivalence check. Migration files must remain independent of evolving ORM models and should not be rewritten after they have been applied to shared environments. A downgrade can reverse schema operations, but cannot necessarily recover deleted data.

## Integration testing

Integration tests use a real PostgreSQL 18 instance provisioned by Testcontainers. They do not use the Compose-managed development database or require `.env.test`, fixed test ports, or manually applied migrations.

With Docker running, execute:

```bash
uv run pytest tests/integration/ -v
```

To run the full backend test suite:

```bash
uv run pytest
```

### Fixture lifecycle

```text
pytest session
├── postgres_container (session scope)
│   └── PostgreSQL 18 container with dynamically allocated host port
├── database_url (session scope)
│   └── Apply Alembic migrations using the generated connection URL
└── individual test
    ├── db_engine (function scope, when requested)
    │   └── AsyncEngine bound to the disposable database
    ├── db_session (function scope, when requested)
    │   ├── Connection and outer transaction
    │   └── AsyncSession using savepoints
    └── db_user (function scope, when requested)
        └── Insert and flush a test user
```

| Fixture | Responsibility |
| --- | --- |
| `postgres_container` | Start a disposable container using `PostgresContainer` from `testcontainers.community.postgres`; stop and remove it during teardown |
| `database_url` | Obtain the generated Psycopg connection URL, apply all migrations, and return the URL |
| `db_engine` | Create an asynchronous engine using the application factory; dispose of it after the test |
| `db_session` | Provide an isolated ORM session bound to a connection with an outer transaction |
| `db_user` | Insert and flush a valid test user for tests that require an existing range owner |

Fixtures are initialized only when requested by a test or another fixture. The container and schema are shared for the pytest session; engines and sessions are created per test. An engine is not reused across independently managed asyncio event loops.

The migration fixture invokes Alembic through a subprocess using the current Python interpreter and the repository root as its working directory. It explicitly supplies the container-generated `DATABASE_URL` to that subprocess without replacing the parent process's environment. Failed migrations abort fixture setup rather than allowing tests to run against an incomplete schema.

### Transaction isolation

Each `db_session` fixture opens a connection and begins an outer transaction. It then binds an `AsyncSession` to that connection with `join_transaction_mode="create_savepoint"`.

```text
Fixture-owned outer transaction
├── Session savepoint
│   ├── INSERT / UPDATE / DELETE
│   └── session.commit() → release savepoint
├── Subsequent session savepoint (if required)
└── Fixture teardown → roll back outer transaction
```

A test can call `session.commit()` without permanently committing the fixture's outer transaction. Closing the session and rolling back that transaction during teardown removes the test's database changes. After an expected integrity error, the test must roll back the failed session transaction before reusing it.

This strategy isolates operations performed through the fixture's connection. It does not roll back independent transactions committed through other connections. Tests requiring real commits, cross-connection write visibility, or concurrent transactions must use a different isolation or cleanup strategy.

### Async event loop compatibility

Asynchronous Psycopg requires a selector-based event loop on Windows. The integration test configuration implements the pytest-asyncio `pytest_asyncio_loop_factories` hook and registers `asyncio.SelectorEventLoop` as the loop factory. This allows pytest-asyncio to manage compatible event loops on Windows and Linux.

The hook configures test execution only; it does not configure the event loop used by the FastAPI application. Verify that the actual API runtime uses a compatible loop on Windows before relying on asynchronous database access there.

### Test coverage

The integration suite exercises:

- Schema creation through Alembic migrations and asynchronous database connectivity.
- ORM user persistence, generated identifiers, and timestamp defaults.
- Transaction isolation when a session commits inside an outer test transaction.
- Saved-range persistence and binary data retrieval from PostgreSQL.
- Unique usernames, foreign keys, mode restrictions, composite uniqueness (including permitted combinations), and required non-null fields.
- Database-level cascade deletion without affecting another user's ranges.
- User repository lookups, flush-without-commit behavior, and uniqueness violations (`test_user_repository.py`).
- Registration persistence, password hashing and offloading, lowercase normalization, duplicate detection, exception propagation, and transaction rollback (`test_user_service.py`).

API tests use the real application with a placeholder database URL and mock the registration service, so they do not verify the full HTTP-to-PostgreSQL path. Savepoint-based service tests do not verify cross-connection commit visibility or concurrent registration requests.

Constraint tests inspect PostgreSQL SQLSTATE codes and, where applicable, constraint or column names to distinguish expected violations from unrelated errors. Binary persistence tests use arbitrary bytes; they do not test a poker range serialization format.

## Continuous integration

The database tests are designed to run within the existing backend GitHub Actions job on an Ubuntu runner. Its `uv run pytest` step provisions PostgreSQL through the same Testcontainers fixtures used locally.

Docker must be available on the runner. No separate PostgreSQL service declaration, test environment file, or development database credentials are required. Testcontainers provisions its own instance, the fixtures apply migrations, and container teardown releases the test infrastructure. The PostgreSQL major version used by the tests matches the development service.

## Key boundaries and current limitations

- **Infrastructure versus business logic:** The `db` package provides schema definitions, database resource factories, and user repositories. Registration is implemented in the service and API layers; authentication and saved-range services are not implemented.
- **Persistent data versus application state:** The database schema covers users and saved ranges only. Application analysis sessions and active equity jobs remain process-local. Authentication session storage is not implemented.
- **Schema versus serialization:** `BYTEA` provides binary storage but does not define or validate the format of a saved poker range.
- **Migrations versus runtime access:** Alembic owns schema evolution; the asynchronous application engine is not responsible for creating tables.
- **Testing versus development:** Integration tests provision their own PostgreSQL database. They must not fall back to the development URL.
- **Runtime integration versus infrastructure readiness:** FastAPI initializes the database engine and injects request-scoped database sessions; the registration service owns its transaction. Production provisioning, database privileges, and operational recovery remain outside the current scope.

See the [backend architecture overview](README.md) for the existing application execution paths and service ownership boundaries.
