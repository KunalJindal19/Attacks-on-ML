"""
run_attack.py
=============
Membership Inference Attack on the NIH Chest X-ray victim models.

Runs BOTH attacks (Baseline + Variance-Enhanced) against BOTH victim models
(overfit + regularized), producing a 4-row comparison table:

  ┌──────────────────────────────────────┬─────────┬──────────────────────────────┐
  │ Victim model                         │ Attack  │  Acc   Prec   Rec    F1      │
  ├──────────────────────────────────────┼─────────┼──────────────────────────────┤
  │ Overfitted (no regularisation)       │ Baseline│  ...   ...    ...    ...     │
  │ Overfitted (no regularisation)       │ Variance│  ...   ...    ...    ...     │
  │ Regularised (Dropout + WD + Aug)     │ Baseline│  ...   ...    ...    ...     │
  │ Regularised (Dropout + WD + Aug)     │ Variance│  ...   ...    ...    ...     │
  └──────────────────────────────────────┴─────────┴──────────────────────────────┘

Pre-conditions
--------------
  Victim_Model/manifest.csv                (run: python Victim_Model/prepare_dataset.py)
  Victim_Model/victim_overfit.pth          (run: python Victim_Model/train_victim.py)
  Victim_Model/victim_regularized.pth      (run: python Victim_Model/train_victim.py --mode regularized)

Usage
-----
  python Membership_Inference_Attack/run_attack.py
  python Membership_Inference_Attack/run_attack.py --victim overfit       # single model
  python Membership_Inference_Attack/run_attack.py --victim regularized   # single model
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import pandas as pd

# ── Resolve paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VICTIM_DIR   = os.path.join(PROJECT_ROOT, "Victim_Model")

sys.path.insert(0, SCRIPT_DIR)   # mia.py, mia_variance.py, shadow_models.py
sys.path.insert(0, VICTIM_DIR)   # api.py

MANIFEST_PATH  = os.path.join(VICTIM_DIR, "manifest.csv")
RESULTS_TXT    = os.path.join(SCRIPT_DIR, "attack_results.txt")
RESULTS_JSON   = os.path.join(SCRIPT_DIR, "attack_results.json")
LOGS_DIR       = os.path.join(SCRIPT_DIR, "logs")

# ── Attack configuration ───────────────────────────────────────────────────────
# NOTE on attacker's data model:
#   The attacker has access to a large UNLABELLED pool of images drawn from the
#   same distribution as the victim's training data — but does NOT know which
#   specific images were used to train the victim.  We simulate this by mixing
#   56k victim-members and 56k non-members, shuffling them, and giving the
#   attacker the combined shuffled pool.  Shadow models are trained on random
#   subsets of this pool, using victim API responses as pseudo-labels.
#
#   Pool sizes are set large (20k member + 20k non-member = 40k) to ensure the
#   shadow models see enough data to mimic the victim's decision boundary.
#   Shadow dataset size (10k/model) is ~18% of the victim's 56k training set —
#   a realistic assumption for a capable attacker.

NUM_SHADOW_MODELS   = 8
SHADOW_DATASET_SIZE = 10_000   # images per shadow model (was 2,500 — too small)
NUM_POOL_MEMBERS    = 20_000   # member images in attacker's pool (was 3,000)
NUM_POOL_NONMEMBERS = 20_000   # non-member images in attacker's pool (was 3,000)
NUM_EVAL_MEMBERS    = 5_000    # held-out evaluation set (was 2,000)
NUM_EVAL_NONMEMBERS = 5_000
RANDOM_SEED         = 42

# Victim model variants — order determines table row order
VICTIM_VARIANTS = [
    {
        "key":        "overfit",
        "label":      "Overfitted (no regularisation)",
        "model_path": os.path.join(VICTIM_DIR, "victim_overfit.pth"),
        "meta_path":  os.path.join(VICTIM_DIR, "victim_overfit_meta.json"),
    },
    {
        "key":        "regularized",
        "label":      "Regularised (Dropout + WD + Aug)",
        "model_path": os.path.join(VICTIM_DIR, "victim_regularized.pth"),
        "meta_path":  os.path.join(VICTIM_DIR, "victim_regularized_meta.json"),
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def print_banner(text: str):
    print(flush=True)
    print("=" * 70, flush=True)
    print(f"  {text}", flush=True)
    print("=" * 70, flush=True)
    sys.stdout.flush()


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: {MANIFEST_PATH} not found. Run prepare_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    member_paths    = df[df["split"] == "member"]["path"].values
    nonmember_paths = df[df["split"] == "nonmember"]["path"].values
    print(f"[DATA] Members:     {len(member_paths)}", flush=True)
    print(f"[DATA] Non-members: {len(nonmember_paths)}", flush=True)
    return member_paths, nonmember_paths


def build_pool_and_eval(member_paths_all, nonmember_paths_all):
    """Return (pool_all, eval_member_paths, eval_nonmember_paths).

    Attacker's pool: a shuffled mix of member + non-member paths.
    The attacker does NOT know which path belongs to which split.
    The pool is drawn first; the remaining paths form the eval set
    (which WE know the membership of, to score the attack).

    Split logic:
      pool_m  = min(NUM_POOL_MEMBERS,  len(members)  - NUM_EVAL_MEMBERS)
      pool_nm = min(NUM_POOL_NONMEMBERS, len(nonmembers) - NUM_EVAL_NONMEMBERS)
    This guarantees NUM_EVAL images are always reserved for evaluation.
    """
    rng = np.random.RandomState(RANDOM_SEED)

    # Reserve evaluation images first, use the rest for the pool
    max_pool_m  = max(0, len(member_paths_all)    - NUM_EVAL_MEMBERS)
    max_pool_nm = max(0, len(nonmember_paths_all) - NUM_EVAL_NONMEMBERS)
    n_pool_m    = min(NUM_POOL_MEMBERS,    max_pool_m)
    n_pool_nm   = min(NUM_POOL_NONMEMBERS, max_pool_nm)

    member_idx    = rng.permutation(len(member_paths_all))
    nonmember_idx = rng.permutation(len(nonmember_paths_all))

    n_eval_m  = min(NUM_EVAL_MEMBERS,    len(member_paths_all)    - n_pool_m)
    n_eval_nm = min(NUM_EVAL_NONMEMBERS, len(nonmember_paths_all) - n_pool_nm)

    pool_member_paths    = member_paths_all[member_idx[:n_pool_m]]
    pool_nonmember_paths = nonmember_paths_all[nonmember_idx[:n_pool_nm]]
    eval_member_paths    = member_paths_all[member_idx[n_pool_m: n_pool_m + n_eval_m]]
    eval_nonmember_paths = nonmember_paths_all[nonmember_idx[n_pool_nm: n_pool_nm + n_eval_nm]]

    # Shuffle the pool so shadow models can't infer membership from order
    pool_all = np.concatenate([pool_member_paths, pool_nonmember_paths])
    rng.shuffle(pool_all)

    print(
        f"\n[SETUP] Pool size:      {len(pool_all)} "
        f"({n_pool_m} member + {n_pool_nm} nonmember, shuffled — attacker cannot distinguish)"
    )
    print(f"[SETUP] Eval members:    {len(eval_member_paths)}")
    print(f"[SETUP] Eval nonmembers: {len(eval_nonmember_paths)}")
    sys.stdout.flush()

    return pool_all, eval_member_paths, eval_nonmember_paths



def confidence_gap_diagnostic(api, eval_member_paths, eval_nonmember_paths):
    """Print and return the mean max-confidence gap between members and non-members."""
    t0 = time.time()
    member_scores    = api.predict(np.array(eval_member_paths,    dtype=object))
    nonmember_scores = api.predict(np.array(eval_nonmember_paths, dtype=object))
    print(
        f"  {len(eval_member_paths) + len(eval_nonmember_paths)} eval images "
        f"queried in {time.time() - t0:.1f}s",
        flush=True,
    )

    member_conf    = np.max(member_scores,    axis=1).mean()
    nonmember_conf = np.max(nonmember_scores, axis=1).mean()
    gap            = member_conf - nonmember_conf

    print(f"\n[DIAG] Mean max-confidence on members:     {member_conf:.4f}")
    print(f"[DIAG] Mean max-confidence on non-members: {nonmember_conf:.4f}")
    print(
        f"[DIAG] Confidence gap (member - nonmember): {gap:+.4f}  "
        f"(positive = MIA signal exists)"
    )
    sys.stdout.flush()

    if gap < 0.05:
        print(
            "\n  WARNING: Confidence gap is very small. "
            "The MIA signal may be weak for this victim.",
            flush=True,
        )
    return gap


# ─── Attack model variants to compare ─────────────────────────────────────────
# Shadow models are trained ONCE; all attack models below reuse the same
# attack dataset.  Add / remove entries here freely.

ATTACK_MODELS = [
    ("Gradient Boosting",   "gradient_boosting",   dict(n_estimators=200, learning_rate=0.05)),
    ("Random Forest",       "random_forest",        dict(n_estimators=200)),
    ("MLP",                 "mlp",                  dict(hidden_layer_sizes=(256, 128), max_iter=500)),
    ("Logistic Regression", "logistic_regression",  dict()),   # linear baseline
]


# ─── Single-variant attack runners ────────────────────────────────────────────

def run_baseline_attack(api, pool_paths, eval_member_paths,
                        eval_nonmember_paths, num_classes: int) -> list[dict]:
    """Train shadow models ONCE, then evaluate all ATTACK_MODELS on the same dataset."""
    from mia import MIA, ModelParameters

    shadow_params = [
        ModelParameters(
            "pytorch_cnn", architecture=arch, num_classes=num_classes,
            epochs=25, batch_size=32, lr=1e-3,
        )
        for arch in ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]
    ]
    mia = MIA(
        victim_model_api=api,
        unlabelled_data=pool_paths,
        num_classes=num_classes,
        num_shadow_models=NUM_SHADOW_MODELS,
        shadow_model_dataset_size=SHADOW_DATASET_SIZE,
        shadow_model_parameters=shadow_params,
    )

    # ── Step 1 + 2: train shadows + build attack dataset (once) ───────────────
    mia.execute_shadow_phase()

    # ── Step 3: evaluate every attack model on the SAME dataset ───────────────
    print("\n[EVAL] Evaluating all attack models on held-out data …", flush=True)
    results = []
    for label, model_type, kwargs in ATTACK_MODELS:
        params  = ModelParameters(model_type, **kwargs)
        metrics = mia.evaluate_attack_model(params, eval_member_paths, eval_nonmember_paths)
        metrics["attack_label"] = f"Baseline / {label}"
        _print_metrics(f"Baseline / {label}", metrics)
        results.append(metrics)

    return results


def run_variance_attack(api, pool_paths, eval_member_paths,
                        eval_nonmember_paths, num_classes: int) -> list[dict]:
    """Train shadow models ONCE, then evaluate all ATTACK_MODELS on the 16-dim dataset."""
    from mia_variance import VarianceMIA
    from mia import ModelParameters

    shadow_params = [
        ModelParameters(
            "pytorch_cnn", architecture=arch, num_classes=num_classes,
            epochs=25, batch_size=32, lr=1e-3,
        )
        for arch in ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]
    ]
    vmia = VarianceMIA(
        victim_model_api=api,
        unlabelled_data=pool_paths,
        num_classes=num_classes,
        num_shadow_models=NUM_SHADOW_MODELS,
        shadow_model_dataset_size=SHADOW_DATASET_SIZE,
        shadow_model_parameters=shadow_params,
    )

    # ── Step 1 + 2: train shadows + build 16-dim attack dataset (once) ────────
    vmia.execute_shadow_phase()

    print("\n  Attack dataset sample (first 3 rows):")
    print(vmia.attack_dataset.head(3).to_string(index=False))

    # ── Step 3: evaluate every attack model on the SAME dataset ───────────────
    print("\n[EVAL] Evaluating all attack models on held-out data …", flush=True)
    results = []
    for label, model_type, kwargs in ATTACK_MODELS:
        params  = ModelParameters(model_type, **kwargs)
        metrics = vmia.evaluate_attack_model(params, eval_member_paths, eval_nonmember_paths)
        metrics["attack_label"] = f"Variance / {label}"
        _print_metrics(f"Variance / {label}", metrics)
        results.append(metrics)

    return results


def _print_metrics(label: str, metrics: dict):
    print(f"\n  {label} Results:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    sys.stdout.flush()


# ─── Per-victim experiment ─────────────────────────────────────────────────────

def run_experiments_for_victim(variant: dict, pool_all, eval_member_paths,
                               eval_nonmember_paths) -> list[dict]:
    """Run baseline + variance attacks against one victim model.

    Shadow models for each attack type are trained ONCE.
    All ATTACK_MODELS are evaluated on the same attack dataset.
    """
    from api import VictimAPI

    model_path = variant["model_path"]
    meta_path  = variant["meta_path"]
    label      = variant["label"]

    if not os.path.exists(model_path):
        print(
            f"\n  SKIPPING {label}: {model_path} not found. "
            f"Run train_victim.py --mode {variant['key']} first.",
            flush=True,
        )
        return []

    with open(meta_path, "r") as f:
        meta = json.load(f)

    num_classes = int(meta["num_classes"])

    print_banner(f"VICTIM: {label.upper()}")
    print(f"  Architecture:    {meta.get('architecture', '?')}")
    print(f"  train_acc:       {meta.get('final_train_acc', 0):.4f}")
    print(f"  val_acc:         {meta.get('final_val_acc', 0):.4f}")
    print(f"  val_AUC:         {meta.get('final_val_auc', 0):.4f}")
    print(f"  Memorization gap:{meta.get('memorization_gap', 0) * 100:+.2f}%")
    print(f"  Dropout:         {meta.get('dropout', False)}")
    print(f"  Weight decay:    {meta.get('weight_decay', 0.0)}")
    print(f"  pos_weight:      {meta.get('pos_weight_used', False)}")
    sys.stdout.flush()

    victim_meta_snapshot = {
        k: v for k, v in meta.items()
        if k not in ("imagenet_mean", "imagenet_std", "label_names")
    }

    results = []

    api = VictimAPI(model_path, num_classes=num_classes, batch_size=32)
    print(f"\n  Inference device: {api.device}", flush=True)

    print("\n  Pre-computing confidence-gap diagnostic …", flush=True)
    gap = confidence_gap_diagnostic(api, eval_member_paths, eval_nonmember_paths)

    # ── Attack 1: Baseline (all ATTACK_MODELS on same shadow dataset) ─────────
    print_banner(f"[{label}] ATTACK 1: STANDARD SHADOW MODEL MIA")
    t0    = time.time()
    rows1 = run_baseline_attack(api, pool_all, eval_member_paths,
                                eval_nonmember_paths, num_classes)
    for r in rows1:
        r["victim_label"] = label
        r["conf_gap"]     = gap
        r["mia_type"]     = "Baseline"
        r["victim_meta"]  = victim_meta_snapshot
    results.extend(rows1)
    print(f"\n  Baseline total runtime: {time.time() - t0:.1f}s", flush=True)

    # ── Attack 2: Variance (all ATTACK_MODELS on same 16-dim dataset) ─────────
    print_banner(f"[{label}] ATTACK 2: VARIANCE-ENHANCED SHADOW MIA")
    t0    = time.time()
    rows2 = run_variance_attack(api, pool_all, eval_member_paths,
                                eval_nonmember_paths, num_classes)
    for r in rows2:
        r["victim_label"] = label
        r["conf_gap"]     = gap
        r["mia_type"]     = "Variance"
        r["victim_meta"]  = victim_meta_snapshot
    results.extend(rows2)
    print(f"\n  Variance total runtime: {time.time() - t0:.1f}s", flush=True)

    return results


def _save_results(all_results: list, total_time: float):
    """Write full results to attack_results.txt (human) and attack_results.json."""
    os.makedirs(LOGS_DIR, exist_ok=True)

    VL = 34
    AL = 38

    # ── Text file ────────────────────────────────────────────────────────────
    with open(RESULTS_TXT, "w") as f:
        f.write("NIH Chest X-ray — MIA Results (Overfit vs Regularized)\n")
        f.write("=" * 90 + "\n\n")
        f.write(
            f"  {'Victim Model':<{VL}}  {'Attack':<{AL}}  "
            f"{'Gap':>7}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}\n"
        )
        f.write("  " + "-" * 88 + "\n")
        for r in all_results:
            f.write(
                f"  {r['victim_label']:<{VL}}  {r['attack_label']:<{AL}}  "
                f"{r['conf_gap']:+7.4f}  {r['accuracy']:7.4f}  "
                f"{r['precision']:7.4f}  {r['recall']:7.4f}  {r['f1']:7.4f}\n"
            )
        f.write(f"\nRandom baseline: 0.5000\n")
        f.write(f"Total runtime:   {total_time:.1f}s\n")

        # Per-victim metadata block
        f.write("\n" + "=" * 90 + "\n")
        f.write("Victim Model Details\n")
        f.write("=" * 90 + "\n")
        seen = set()
        for r in all_results:
            vl = r["victim_label"]
            if vl not in seen:
                seen.add(vl)
                f.write(f"\n  {vl}\n")
                vm = r.get("victim_meta", {})
                for k, v in vm.items():
                    f.write(f"    {k:<25s}: {v}\n")

    # ── JSON file ────────────────────────────────────────────────────────────
    with open(RESULTS_JSON, "w") as f:
        json.dump(
            {
                "total_runtime_s": round(total_time, 2),
                "random_baseline": 0.5,
                "results": all_results,
            },
            f, indent=2,
        )

    print(f"\n  Results (TXT): {RESULTS_TXT}", flush=True)
    print(f"  Results (JSON): {RESULTS_JSON}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run MIA against overfitted and/or regularized victim models"
    )
    parser.add_argument(
        "--victim", type=str, default="both",
        choices=["both", "overfit", "regularized"],
        help="Which victim model(s) to attack. Default: both",
    )
    args = parser.parse_args()

    print_banner("Shadow Model MIA — NIH Chest X-ray")

    # 1. Load manifest (shared by both victims)
    print("\n[SETUP] Loading manifest …", flush=True)
    member_paths_all, nonmember_paths_all = load_manifest()

    # 2. Build shared pool + eval split (same random seed → fair comparison)
    pool_all, eval_member_paths, eval_nonmember_paths = build_pool_and_eval(
        member_paths_all, nonmember_paths_all
    )

    # 3. Determine which victims to run
    if args.victim == "both":
        variants = VICTIM_VARIANTS
    else:
        variants = [v for v in VICTIM_VARIANTS if v["key"] == args.victim]

    # 4. Run experiments
    grand_start  = time.time()
    all_results  = []

    for variant in variants:
        results = run_experiments_for_victim(
            variant, pool_all, eval_member_paths, eval_nonmember_paths
        )
        all_results.extend(results)
        # Incremental save after each victim so a crash doesn't lose completed results
        if all_results:
            _save_results(all_results, time.time() - grand_start)

    total_time = time.time() - grand_start

    if not all_results:
        print("\nNo results — did you train the victim models first?", flush=True)
        return

    # 5. Final comparison table
    print_banner("FINAL COMPARISON TABLE")

    VL = 34   # victim label width
    AL = 38   # attack label width

    header = (
        f"\n  {'Victim Model':<{VL}}  {'Attack':<{AL}}  "
        f"{'Gap':>7}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}"
    )
    print(header)
    print("  " + "-" * 88)

    for r in all_results:
        print(
            f"  {r['victim_label']:<{VL}}  {r['attack_label']:<{AL}}  "
            f"{r['conf_gap']:+7.4f}  {r['accuracy']:7.4f}  "
            f"{r['precision']:7.4f}  {r['recall']:7.4f}  {r['f1']:7.4f}"
        )

    print(f"\n  Random baseline: 0.5000")
    print(f"  Total runtime:   {total_time:.1f}s")
    sys.stdout.flush()

    # 6. Final save
    _save_results(all_results, total_time)


if __name__ == "__main__":
    main()
