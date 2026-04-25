# CV RAG backend

FastAPI service for answering questions over resume markdown files.

## Local run

```bash
cp cv-rag/.env.example cv-rag/.env
make install
make run
```

## Checks

```bash
make lint
make typecheck
make test
```

## Docker

```bash
docker compose up -d --build
```

Required runtime variable:

- `OPENROUTER__API_KEY`

Health endpoints:

- `/health` - process is alive
- `/ready` - index and OpenRouter config are ready
