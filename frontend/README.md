# PyTorchi: Ore analyzer frontend

React + Vite интерфейс для локального сервиса анализа руды.

```bash
npm install
npm run dev -- --host 0.0.0.0
```

Vite проксирует `/api` на `http://localhost:8000`. Для другого адреса:

```bash
VITE_API_PROXY_TARGET=http://backend:8000 npm run dev -- --host 0.0.0.0
```

Production build:

```bash
npm run test
npm run build
```

Результат создаётся в `dist/`. В production запросы идут на same-origin `/api`;
переопределить base URL можно через `VITE_API_BASE`.

Встроенный пример открывается только явной кнопкой и всегда помечен как демо.
