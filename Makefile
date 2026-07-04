.PHONY: check config build up down logs dev dev-down gpu-up clean

check:
	sh scripts/check-infra.sh

config:
	docker compose config --quiet

build:
	docker compose build

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

dev:
	docker compose -f docker-compose.dev.yml up --build

dev-down:
	docker compose -f docker-compose.dev.yml down

gpu-up:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d

clean:
	docker compose down --remove-orphans
	docker compose -f docker-compose.dev.yml down --remove-orphans
