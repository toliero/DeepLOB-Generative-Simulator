# DeepLOB — Generative Diffusion for Synthetic Market Microstructure

An academic simulator for **synthetic market path generation** and **stylized-fact validation**.

The intent is to provide a controlled environment for:
- non-Markovian dynamics experiments,
- rough-volatility toggles,
- systemic coupling stress tests,
- and reproducible “paper-style” diagnostics (plots + JSON report).

## Core idea (high level)

The orchestration script (`research_main.py`) coordinates:
- data retrieval and preprocessing,
- a diffusion-style generative pipeline (PyTorch),
- and evaluation of stylized facts (distributional tests, volatility clustering, etc.).

## Outputs

Written to the repo root for simplicity:
- `market_dashboard.png` — Matplotlib dashboard of synthetic vs real diagnostics
- `experiment_report.json` — structured experiment metadata
- `evaluation_results.png` — additional evaluation visuals

## Repository structure

This repository contains **only** the Python research stack (no web UI, Next.js, or portfolio pages).

```
.
├── research_main.py    # main orchestrator (CLI)
├── diffusion_model.py  # model architecture / generation
├── data_processing.py  # data retrieval + feature pipeline (yfinance-based)
├── evaluation.py       # evaluation suite + plots
├── requirements.txt
└── README.md
```

## Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# baseline run
python3 research_main.py --coupling 0.2

# rough volatility regime (example)
python3 research_main.py --rough_vol --hurst 0.10 --coupling 0.35
```

## Notes

- **PyTorch is required** for the diffusion components. If you don’t have it installed yet,
  install via your preferred method (pip/conda) compatible with your machine.
- `yfinance` intraday data availability is limited; the pipeline degrades gracefully to the maximum available window.

