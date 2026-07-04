#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/docker-compose.yml" ]]; then
  PROJECT_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../docker-compose.yml" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  echo "Ошибка: не могу найти docker-compose.yml рядом со скриптом или на уровень выше" >&2
  exit 1
fi

ENV_FILE="${PROJECT_ROOT}/.env"
CPU_COMPOSE=(-f docker-compose.yml)
CUDA_COMPOSE=(-f docker-compose.yml -f docker-compose.gpu.yml)

BOLD="$(tput bold 2>/dev/null || true)"
DIM="$(tput dim 2>/dev/null || true)"
RESET="$(tput sgr0 2>/dev/null || true)"
GREEN="$(tput setaf 2 2>/dev/null || true)"
YELLOW="$(tput setaf 3 2>/dev/null || true)"
BLUE="$(tput setaf 4 2>/dev/null || true)"
RED="$(tput setaf 1 2>/dev/null || true)"
CYAN="$(tput setaf 6 2>/dev/null || true)"

print_header() {
  clear > /dev/tty
  cat > /dev/tty <<EOF
${CYAN}${BOLD}
╔══════════════════════════════════════════════════════╗
║              PyTorchi: Ore analyzer                 ║
║              Docker launch wizard                   ║
╚══════════════════════════════════════════════════════╝
${RESET}
EOF
}

die() {
  echo "${RED}Ошибка:${RESET} $*" >&2
  exit 1
}

info() {
  echo "${BLUE}●${RESET} $*"
}

success() {
  echo "${GREEN}✓${RESET} $*"
}

warn() {
  echo "${YELLOW}!${RESET} $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "команда '$1' не найдена"
}

is_macos() {
  [[ "$(uname -s)" == "Darwin" ]]
}

is_linux() {
  [[ "$(uname -s)" == "Linux" ]]
}

read_key() {
  local key
  IFS= read -rsn1 key < /dev/tty || true

  if [[ "$key" == $'\x1b' ]]; then
    IFS= read -rsn2 -t 0.1 key < /dev/tty || true
    case "$key" in
      "[A") echo "up" ;;
      "[B") echo "down" ;;
      *) echo "esc" ;;
    esac
  elif [[ "$key" == "" ]]; then
    echo "enter"
  else
    echo "$key"
  fi
}

select_menu() {
  local title="$1"
  shift
  local options=("$@")
  local selected=0
  local key

  while true; do
    clear > /dev/tty

    cat > /dev/tty <<EOF
${CYAN}${BOLD}
╔══════════════════════════════════════════════════════╗
║              PyTorchi: Ore analyzer                 ║
║              Docker launch wizard                   ║
╚══════════════════════════════════════════════════════╝
${RESET}
EOF

    {
      echo "${BOLD}${title}${RESET}"
      echo "${DIM}Используй ↑/↓ и Enter${RESET}"
      echo

      for i in "${!options[@]}"; do
        if [[ "$i" -eq "$selected" ]]; then
          echo "  ${GREEN}➜ ${BOLD}${options[$i]}${RESET}"
        else
          echo "    ${options[$i]}"
        fi
      done
    } > /dev/tty

    key="$(read_key)"

    case "$key" in
      up)
        ((selected--)) || true
        if (( selected < 0 )); then
          selected=$((${#options[@]} - 1))
        fi
        ;;
      down)
        ((selected++)) || true
        if (( selected >= ${#options[@]} )); then
          selected=0
        fi
        ;;
      enter)
        printf "%s\n" "$selected"
        return 0
        ;;
    esac
  done
}

prompt_default() {
  local label="$1"
  local default="$2"
  local value

  printf "%s%s%s %s[%s]%s: " "$BOLD" "$label" "$RESET" "$DIM" "$default" "$RESET" > /dev/tty
  IFS= read -r value < /dev/tty

  if [[ -z "$value" ]]; then
    printf "%s\n" "$default"
  else
    printf "%s\n" "$value"
  fi
}

validate_port() {
  local port="$1"

  [[ "$port" =~ ^[0-9]+$ ]] || return 1
  (( port >= 1 && port <= 65535 )) || return 1

  return 0
}

env_get() {
  local key="$1"
  local default="$2"

  if [[ -f "$ENV_FILE" ]]; then
    local value
    value="$(grep -E "^${key}=" "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"
    if [[ -n "$value" ]]; then
      echo "$value"
      return 0
    fi
  fi

  echo "$default"
}

env_set() {
  local key="$1"
  local value="$2"

  touch "$ENV_FILE"

  if grep -qE "^${key}=" "$ENV_FILE"; then
    # macOS/BSD sed and GNU sed compatible replacement.
    local escaped
    escaped="$(printf '%s' "$value" | sed 's/[\/&]/\\&/g')"

    if sed --version >/dev/null 2>&1; then
      sed -i "s/^${key}=.*/${key}=${escaped}/" "$ENV_FILE"
    else
      sed -i "" "s/^${key}=.*/${key}=${escaped}/" "$ENV_FILE"
    fi
  else
    printf '\n%s=%s\n' "$key" "$value" >> "$ENV_FILE"
  fi
}

check_cuda_hint() {
  if is_macos; then
    warn "Ты на macOS. CUDA-контейнер обычно не сможет использовать NVIDIA GPU через Docker Desktop."
  elif is_linux; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      warn "nvidia-smi не найден. Проверь драйвер NVIDIA и NVIDIA Container Toolkit."
    else
      success "NVIDIA GPU обнаружена:"
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    fi
  else
    warn "CUDA-режим требует корректного NVIDIA passthrough в Docker."
  fi

  echo
}

main() {
  cd "$PROJECT_ROOT"

  require_cmd docker

  if ! docker compose version >/dev/null 2>&1; then
    die "Docker Compose v2 не найден. Нужна команда: docker compose ..."
  fi

  print_header

  [[ -f "docker-compose.yml" ]] || die "docker-compose.yml не найден в ${PROJECT_ROOT}"
  [[ -f "docker-compose.gpu.yml" ]] || warn "docker-compose.gpu.yml не найден, CUDA-режим будет недоступен"

  local default_host default_port default_model_dir
  local default_talc_ckpt default_sulfide_ckpt
  local mode_index mode mode_device
  local app_host app_port model_dir talc_ckpt sulfide_ckpt
  local compose_args=()

  default_host="$(env_get APP_HOST "127.0.0.1")"
  default_port="$(env_get APP_PORT "8080")"
  default_model_dir="$(env_get MODEL_DIR "${HOME}/models")"
  default_talc_ckpt="$(env_get TALC_CHECKPOINT_FILE "talc.pt")"
  default_sulfide_ckpt="$(env_get SULFIDE_CHECKPOINT_FILE "sulfide.pt")"

  mode_index="$(select_menu "Выбери режим inference:" "CPU" "CUDA / NVIDIA GPU")"

  if [[ "$mode_index" == "0" ]]; then
    mode="CPU"
    mode_device="cpu"
    compose_args=("${CPU_COMPOSE[@]}")
  else
    mode="CUDA"
    mode_device="cuda"
    compose_args=("${CUDA_COMPOSE[@]}")
  fi

  print_header
  echo "${BOLD}Настройки запуска${RESET}"
  echo

  app_host="$(prompt_default "APP_HOST, 127.0.0.1 только для этого компьютера, 0.0.0.0 для локальной сети" "$default_host")"

  while true; do
    app_port="$(prompt_default "APP_PORT" "$default_port")"
    if validate_port "$app_port"; then
      break
    fi
    warn "Порт должен быть числом от 1 до 65535."
  done

  model_dir="$(prompt_default "MODEL_DIR, абсолютный путь к папке с весами" "$default_model_dir")"
  talc_ckpt="$(prompt_default "TALC_CHECKPOINT_FILE" "$default_talc_ckpt")"
  sulfide_ckpt="$(prompt_default "SULFIDE_CHECKPOINT_FILE" "$default_sulfide_ckpt")"

  print_header
  echo "${BOLD}Проверка настроек${RESET}"
  echo

  info "Режим: ${mode}"
  info "Host: ${app_host}"
  info "Port: ${app_port}"
  info "Model dir: ${model_dir}"
  info "Talc checkpoint: ${talc_ckpt}"
  info "Sulfide checkpoint: ${sulfide_ckpt}"
  echo

  if [[ "$mode" == "CUDA" ]]; then
    check_cuda_hint
  fi

  if [[ ! -d "$model_dir" ]]; then
    warn "MODEL_DIR не существует: ${model_dir}"
    warn "Приложение может запуститься, но анализ будет недоступен до появления весов."
    echo
  else
    if [[ ! -f "${model_dir}/${talc_ckpt}" ]]; then
      warn "Не найден файл: ${model_dir}/${talc_ckpt}"
    fi
    if [[ ! -f "${model_dir}/${sulfide_ckpt}" ]]; then
      warn "Не найден файл: ${model_dir}/${sulfide_ckpt}"
    fi
    echo
  fi

  if [[ -f "$ENV_FILE" ]]; then
    local backup="${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$ENV_FILE" "$backup"
    success "Сделал backup .env: ${backup}"
  fi

  env_set APP_HOST "$app_host"
  env_set APP_PORT "$app_port"
  env_set MODEL_DIR "$model_dir"
  env_set TALC_CHECKPOINT_FILE "$talc_ckpt"
  env_set SULFIDE_CHECKPOINT_FILE "$sulfide_ckpt"
  env_set MODEL_DEVICE "$mode_device"

  success ".env обновлён"
  echo

  info "Проверяю docker compose config..."
  docker compose "${compose_args[@]}" config --quiet
  success "Compose-конфигурация валидна"
  echo

  info "Собираю и запускаю контейнеры..."
  docker compose "${compose_args[@]}" up --build -d
  echo

  success "Готово!"
  echo
  echo "${BOLD}Открыть:${RESET} http://localhost:${app_port}"
  if [[ "$app_host" == "0.0.0.0" ]]; then
    echo "${BOLD}В локальной сети:${RESET} http://<IP_этого_компьютера>:${app_port}"
  fi
  echo
  echo "${DIM}Полезные команды:${RESET}"
  echo "  docker compose ${compose_args[*]} ps"
  echo "  docker compose ${compose_args[*]} logs -f"
  echo "  docker compose ${compose_args[*]} down"
}

main "$@"