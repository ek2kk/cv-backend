UV=uv
.PHONY: install run test lint fmt typecheck check build up down logs

install:
	$(info "Installing project and dev dependencies")
	$(UV) sync

run:
	$(info "Running API locally with the canonical command")
	$(UV) run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(info "Running smoke tests")
	$(UV) run pytest

lint:
	$(info "Running Ruff checks")
	$(UV) run ruff check .

fmt:
	$(info "Formatting code with Ruff")
	$(UV) run ruff format .

typecheck:
	$(info "Running mypy")
	$(UV) run mypy app

check:
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

build:
	$(info "Building Docker image")
	docker compose -f docker-compose.yml build

up:
	$(info "Run Docker image")
	docker compose -f docker-compose.yml up -d

down:
	$(info "Stopping Docker image")
	docker compose -f docker-compose.yml down

logs:
	$(info "Show logs Docker")
	docker compose logs -f --tail 300
