# Repository Guidelines

This repository hosts a multi-component quantitative trading system. Most code lives under `quant_system_full/` with Python services, a React UI, and support scripts.

## Project Structure & Module Organization
- `quant_system_full/` — core Python system and scripts.
  - `bot/` — bot logic and automation (`requirements.txt`).
  - `dashboard/{backend,frontend,worker}/` — monitoring/dashboard services (each with `requirements.txt`).
  - `UI/` — React UI (`package.json`).
  - `yahoo-finance-server/` — market data microservice (Python, `pyproject.toml`).
  - `scripts/` — shared scripts (`requirements.txt`).
  - Test files at root: `test_*.py` (e.g., `test_ultra_ai_api.py`).
  - Artifacts: `reports/`, `data_cache/`, `gpu_models/`, and various CSV/JSON inputs.

## Build, Test, and Development Commands
- Python env (Windows example):
  - `cd quant_system_full`
  - `python -m venv .venv && .\.venv\Scripts\activate`
  - `pip install -r scripts/requirements.txt`
  - Install per-component deps as needed: `pip install -r dashboard\backend\requirements.txt` (repeat for `bot`, `dashboard\worker`, etc.).
- Run core system: `python start_system.py` or `python run_ml_trading_system.py`
- Start all helpers (optional): `start_all.bat`, `start_react_ui.bat`
- UI (from `quant_system_full\UI`): `npm install && npm run dev`
- Tests (pytest): `cd quant_system_full && pytest -q`

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent, `snake_case` for modules/functions, `CapWords` for classes, prefer type hints. Keep functions focused and documented with short docstrings.
- JS/TS (UI): follow project scripts; use Prettier/ESLint if configured in `UI/`.
- Filenames: tests as `test_*.py`; configs as `*.json` / `*.env` examples.

## Testing Guidelines
- Framework: `pytest` with test files named `test_*.py` in `quant_system_full/`.
- Add targeted unit tests alongside affected modules or in root as consistent with existing patterns.
- Run locally: `pytest -q`; include a brief test plan in PRs (commands + expected output).

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope. Example: `feat(bot): add retry on API throttle)`.
- PRs: clear summary, linked issues, steps to validate (pytest output, relevant logs), screenshots for UI changes, and updated docs when applicable.

## Security & Configuration Tips
- Do not commit secrets (`.env`, keys, tokens). Use `quant_system_full/.env` locally; start from `config.example.env`.
- GPU/CUDA: if required, use `set_cuda_env.bat` and `gpu_config.json` as templates; document hardware assumptions.

## Agent-Specific Instructions
- Keep changes minimal and scoped; do not refactor unrelated code.
- Place new files in the most specific module directory; follow naming and testing patterns above.
