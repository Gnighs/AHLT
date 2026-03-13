#! /usr/bin/python3
#
# Grid search for CRF hyperparameters
# Run from the same directory as run.py:
#   python3 grid_search.py
#
# Results are printed to stdout and saved to grid_search_results.txt

import sys, os, itertools, subprocess

# ── paths ────────────────────────────────────────────────────────────────────
import paths  # reuse existing paths module

TRAIN_FEAT  = os.path.join(paths.PREPROCESS, "train.feat")
DEVEL_FEAT  = os.path.join(paths.PREPROCESS, "devel.feat")
DEVEL_XML   = os.path.join(paths.DATA,       "devel.xml")
MODEL_FILE  = os.path.join(paths.MODELS,     "model.CRF")
RESULTS_OUT = os.path.join(paths.RESULTS,    "devel-CRF.out")
STATS_FILE  = os.path.join(paths.RESULTS,    "devel-CRF.stats")
OUTPUT_LOG  = "grid_search_results.txt"

os.makedirs(paths.MODELS,  exist_ok=True)
os.makedirs(paths.RESULTS, exist_ok=True)

# ── hyperparameter grid ──────────────────────────────────────────────────────
# Add or remove values freely — every combination will be tried
GRID = {
    "c1":             [0.01, 0.05, 0.1],
    "c2":             [0.2,  0.5,  1.0],
    "max_iterations": [100, 200],
}

# ── helpers ──────────────────────────────────────────────────────────────────
def parse_stats(stats_file):
    """Extract macro F1 and micro F1 from the .stats file."""
    macro_f1 = micro_f1 = None
    try:
        with open(stats_file) as f:
            for line in f:
                if "M.avg" in line:
                    parts = line.split()
                    macro_f1 = float(parts[-1].replace('%',''))
                elif "m.avg" in line and "no class" not in line:
                    parts = line.split()
                    micro_f1 = float(parts[-1].replace('%',''))
    except FileNotFoundError:
        pass
    return macro_f1, micro_f1


def run_experiment(params):
    """Train and evaluate one CRF configuration. Returns (macro_f1, micro_f1)."""
    # ── train ────────────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import train
    train(TRAIN_FEAT, params, MODEL_FILE)

    # ── predict ──────────────────────────────────────────────────────────────
    from predict import predict
    predict(DEVEL_FEAT, MODEL_FILE, RESULTS_OUT)

    # ── evaluate ─────────────────────────────────────────────────────────────
    sys.path.append(paths.UTIL)
    from evaluator import evaluate
    evaluate("NER", DEVEL_XML, RESULTS_OUT, STATS_FILE)

    return parse_stats(STATS_FILE)


# ── grid search ──────────────────────────────────────────────────────────────
keys   = list(GRID.keys())
values = list(GRID.values())
combos = list(itertools.product(*values))

print(f"Running grid search: {len(combos)} combinations\n")
print(f"{'c1':>8} {'c2':>6} {'max_iter':>10}  {'MacroF1':>9} {'MicroF1':>9}")
print("-" * 55)

best_macro = -1
best_params = None
results = []

for combo in combos:
    params = dict(zip(keys, combo))
    macro, micro = run_experiment(params)

    tag = " ← best" if macro and macro > best_macro else ""
    if macro and macro > best_macro:
        best_macro  = macro
        best_params = params.copy()

    line = (f"c1={params['c1']:<6} c2={params['c2']:<5} "
            f"iter={params['max_iterations']:<5}  "
            f"macro={macro or 'ERR':>6}  micro={micro or 'ERR':>6}{tag}")
    print(line)
    results.append(line)

# ── summary ──────────────────────────────────────────────────────────────────
summary = [
    "\n" + "="*55,
    f"Best macro F1 : {best_macro:.1f}%",
    f"Best params   : {best_params}",
    "="*55,
]
for s in summary:
    print(s)
results.extend(summary)

with open(OUTPUT_LOG, "w") as f:
    f.write("\n".join(results))

print(f"\nFull results saved to {OUTPUT_LOG}")