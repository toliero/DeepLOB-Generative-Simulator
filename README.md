# Lab Simulators

High-fidelity quantitative finance simulation “lab” (diffusion-based market path generation + stylized-facts dashboard).

## Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 research_main.py --coupling 0.2
```

Outputs (written to repo root):
- `market_dashboard.png`
- `experiment_report.json`
- `evaluation_results.png`

