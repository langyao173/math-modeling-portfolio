# -*- coding: utf-8 -*-
"""Shared audit configuration for the APMCM B solution scripts."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BASE_DIR / "代码"
FIG_DIR = BASE_DIR / "图表"
DATA_FILE = BASE_DIR / "附件2_data.tsv"

beta_min = 0.1
beta_max = 0.3
gamma_min = 3.0
gamma_max = 4.5
N_levels = [2, 4, 6, 8, 10]
design_star = (0.22, 4.5, 4)

seed_list = list(range(10))
random_seed = 0

sobol_n_base = 32768
sobol_original_n_base = 8192
mc_n = 20000
strict_cv = True
domain_clip = False

grid_specs = [
    {"name": "coarse", "beta_step": 0.005, "gamma_step": 0.05},
    {"name": "current", "beta_step": 0.0025, "gamma_step": 0.025},
    {"name": "fine", "beta_step": 0.00125, "gamma_step": 0.0125},
]

cv_inner_splits = 5
gpr_restarts_cv = int(os.environ.get("APMCM_GPR_RESTARTS_CV", "2"))
gpr_restarts_final = int(os.environ.get("APMCM_GPR_RESTARTS_FINAL", "5"))

SIGMA_BETA_REL = 0.02
SIGMA_GAMMA_REL = 0.015

TARGETS = ["R", "dP", "phi"]
INPUTS = ["beta", "gamma", "N"]

def ensure_dirs() -> None:
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
