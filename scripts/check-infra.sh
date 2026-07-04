#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

command -v docker >/dev/null 2>&1 || {
  echo "docker is required" >&2
  exit 1
}

test -f backend/requirements.lock
test -f backend/vendor/talk_combined/talc_analysis/__init__.py
test -f backend/vendor/talk_sulfid/ore_classifier/__init__.py
test -f frontend/package.json
test -f nginx.conf

docker compose config --quiet
docker compose -f docker-compose.dev.yml config --quiet
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  config --quiet

git diff --check
echo "Infrastructure checks passed."
