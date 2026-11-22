# World Cup Forecasting

Reproducible Python implementation for 2026 APMCM Problem A. The project preserves the exact official linear/logistic baselines and adds leakage-safe joint-goal modeling, time-ordered validation, uncertainty diagnostics, fitted-state prediction certificates, and an independently verified manifest. MATLAB R2026a is an optional experiment/parity workbench and is never part of the production dependency chain.

The frozen final system runs with:

```powershell
uv run python main.py
```

For a conventional clean-room environment, `pip install -r requirements.txt` is also supported.

Research selection is deliberately separate:

```powershell
uv run python main.py --config research --output research_outputs
```

See `doc/README.md` for execution, `doc/RESEARCH_REPORT.md` for the frozen seven-fold study,
one-shot adverse 2014 audit, and scenario-only 2026 evidence, `doc/AUDIT.md` for verification, and
`doc/MODEL_CARD.md` for intended use and limitations.
