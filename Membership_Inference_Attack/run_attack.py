"""
run_attack.py
=============
Membership Inference Attack on the NIH Chest X-ray victim model.

Runs two attack variants:
  1. Baseline Shadow Model MIA
  2. Variance-Enhanced Shadow Model MIA

Pre-conditions
--------------
  Victim_Model/manifest.csv      (run: python Victim_Model/prepare_dataset.py)
  Victim_Model/victim.pth        (run: python Victim_Model/train_victim.py)
  Victim_Model/victim_meta.json  (created automatically by train_victim.py)

Usage
-----
  python Membership_Inference_Attack/run_attack.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# ── Resolve paths so the script can be run from any working directory ─────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VICTIM_DIR   = os.path.join(PROJECT_ROOT, "Victim_Model")

# Add Membership_Inference_Attack/ to path so mia.py / shadow_models.py resolve
sys.path.insert(0, SCRIPT_DIR)
# Add Victim_Model/ to path so api.py resolves
sys.path.insert(0, VICTIM_DIR)

MANIFEST_PATH = os.path.join(VICTIM_DIR, "manifest.csv")
MODEL_PATH    = os.path.join(VICTIM_DIR, "victim.pth")
META_PATH     = os.path.join(VICTIM_DIR, "victim_meta.json")
RESULTS_PATH  = os.path.join(SCRIPT_DIR, "attack_results.txt")

# ── Attack configuration (matches prompt recommendations) ────────────────────

NUM_SHADOW_MODELS   = 8
SHADOW_DATASET_SIZE = 2_500    # images per shadow model
NUM_POOL_MEMBERS    = 3_000    # from member paths → attacker's pool
NUM_POOL_NONMEMBERS = 3_000    # from non-member paths → attacker's pool
NUM_EVAL_MEMBERS    = 2_000    # held-out for evaluation
NUM_EVAL_NONMEMBERS = 2_000
RANDOM_SEED         = 42


# ─── Helpers ──────────────────────────────────────────────────────────────────

def print_banner(text: str):
    print(flush=True)
    print("=" * 70, flush=True)
    print(f"  {text}", flush=True)
    print("=" * 70, flush=True)
    sys.stdout.flush()


def load_manifest():
    """Load manifest and return (member_paths, nonmember_paths) as np arrays."""
    for path, label in [(MANIFEST_PATH, "manifest.csv"),
                         (MODEL_PATH,    "victim.pth"),
                         (META_PATH,     "victim_meta.json")]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run the prerequisite script first.")
            sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    member_paths    = df[df["split"] == "member"]["path"].values
    nonmember_paths = df[df["split"] == "nonmember"]["path"].values

    print(f"[DATA] Members:     {len(member_paths)}", flush=True)
    print(f"[DATA] Non-members: {len(nonmember_paths)}", flush=True)
    return member_paths, nonmember_paths


# ─── Attack 1: Baseline Shadow MIA ────────────────────────────────────────────

def run_baseline_attack(api, pool_paths, eval_member_paths,
                        eval_nonmember_paths, num_classes: int) -> dict:
    from mia import MIA, ModelParameters

    print_banner("ATTACK 1: STANDARD SHADOW MODEL MIA (BASELINE)")

    # Cycle through three lightweight CNN architectures for diversity
    shadow_archs = ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]
    shadow_params = [
        ModelParameters(
            "pytorch_cnn",
            architecture=arch,
            num_classes=num_classes,
            epochs=15,
            batch_size=32,
            lr=1e-3,
        )
        for arch in shadow_archs
    ]

    mia = MIA(
        victim_model_api=api,
        unlabelled_data=pool_paths,
        num_classes=num_classes,
        num_shadow_models=NUM_SHADOW_MODELS,
        shadow_model_dataset_size=SHADOW_DATASET_SIZE,
        shadow_model_parameters=shadow_params,
        attack_model_parameters=ModelParameters(
            "gradient_boosting", n_estimators=100, learning_rate=0.1
        ),
    )
    mia.execute()

    print("\n[STEP 2] Evaluating on held-out data …", flush=True)
    metrics = mia.evaluate(eval_member_paths, eval_nonmember_paths)

    print(f"\n  Baseline Shadow MIA Results:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    sys.stdout.flush()

    return {
        "Model":     "Baseline Shadow Model (Gradient Boosting)",
        "Accuracy":  metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall":    metrics["recall"],
        "F1":        metrics["f1"],
    }


# ─── Attack 2: Variance-Enhanced Shadow MIA ───────────────────────────────────

def run_variance_attack(api, pool_paths, eval_member_paths,
                        eval_nonmember_paths, num_classes: int) -> dict:
    from mia_variance import VarianceMIA
    from mia import ModelParameters

    print_banner("ATTACK 2: VARIANCE-ENHANCED SHADOW MODEL MIA")

    shadow_archs = ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]
    shadow_params = [
        ModelParameters(
            "pytorch_cnn",
            architecture=arch,
            num_classes=num_classes,
            epochs=15,
            batch_size=32,
            lr=1e-3,
        )
        for arch in shadow_archs
    ]

    vmia = VarianceMIA(
        victim_model_api=api,
        unlabelled_data=pool_paths,
        num_classes=num_classes,
        num_shadow_models=NUM_SHADOW_MODELS,
        shadow_model_dataset_size=SHADOW_DATASET_SIZE,
        shadow_model_parameters=shadow_params,
        attack_model_parameters=ModelParameters(
            "gradient_boosting", n_estimators=100, learning_rate=0.1
        ),
    )
    vmia.execute()

    # Quick peek at the attack dataset to verify variance feature
    print("\n  Attack dataset sample (first 3 rows):")
    print(vmia.attack_dataset.head(3).to_string(index=False))
    sys.stdout.flush()

    print("\n[STEP 2] Evaluating on held-out data …", flush=True)
    metrics = vmia.evaluate(eval_member_paths, eval_nonmember_paths)

    print(f"\n  VarianceMIA Results:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    sys.stdout.flush()

    return {
        "Model":     "Variance Shadow Model (Gradient Boosting)",
        "Accuracy":  metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall":    metrics["recall"],
        "F1":        metrics["f1"],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print_banner("Shadow Model MIA on NIH Chest X-ray Victim Model")

    # 1. Load manifest
    print("\n[SETUP] Loading manifest …", flush=True)
    member_paths_all, nonmember_paths_all = load_manifest()

    # 2. Load victim metadata
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    num_classes = int(meta["num_classes"])
    print(f"[SETUP] Architecture:  {meta.get('architecture', 'unknown')}")
    print(f"[SETUP] Num classes:   {num_classes}")
    print(f"[SETUP] Label names:   {meta.get('label_names', [])}")
    print(
        f"[SETUP] Victim  train_acc={meta.get('final_train_acc', 0):.4f}  "
        f"val_acc={meta.get('final_val_acc', 0):.4f}  "
        f"gap={(meta.get('memorization_gap', 0)) * 100:+.2f}%"
    )
    sys.stdout.flush()

    # 3. Build pool (attacker's view: mixed members + non-members, shuffled)
    #    and held-out eval set (we know ground truth for measuring attack perf)
    rng = np.random.RandomState(RANDOM_SEED)

    # Cap pool size to at most 66% of available paths, leave rest for eval
    max_pool_m  = int(len(member_paths_all)    * 0.66)
    max_pool_nm = int(len(nonmember_paths_all) * 0.66)
    n_pool_m    = min(NUM_POOL_MEMBERS,    max_pool_m)
    n_pool_nm   = min(NUM_POOL_NONMEMBERS, max_pool_nm)

    member_idx    = rng.permutation(len(member_paths_all))
    nonmember_idx = rng.permutation(len(nonmember_paths_all))

    pool_m_idx  = member_idx[:n_pool_m]
    pool_nm_idx = nonmember_idx[:n_pool_nm]

    n_eval_m  = min(NUM_EVAL_MEMBERS,    len(member_paths_all)    - n_pool_m)
    n_eval_nm = min(NUM_EVAL_NONMEMBERS, len(nonmember_paths_all) - n_pool_nm)
    eval_m_idx  = member_idx[n_pool_m: n_pool_m + n_eval_m]
    eval_nm_idx = nonmember_idx[n_pool_nm: n_pool_nm + n_eval_nm]

    pool_member_paths    = member_paths_all[pool_m_idx]
    pool_nonmember_paths = nonmember_paths_all[pool_nm_idx]
    eval_member_paths    = member_paths_all[eval_m_idx]
    eval_nonmember_paths = nonmember_paths_all[eval_nm_idx]

    # Attacker's pool: mix and shuffle so membership is unknown
    pool_all = np.concatenate([pool_member_paths, pool_nonmember_paths])
    rng.shuffle(pool_all)

    print(
        f"\n[SETUP] Pool size:      {len(pool_all)} "
        f"({n_pool_m} member + {n_pool_nm} nonmember, shuffled)"
    )
    print(f"[SETUP] Eval members:    {len(eval_member_paths)}")
    print(f"[SETUP] Eval nonmembers: {len(eval_nonmember_paths)}")
    sys.stdout.flush()

    # 4. Load victim API
    print(f"\n[SETUP] Loading victim model …", flush=True)
    from api import VictimAPI
    api = VictimAPI(MODEL_PATH, num_classes=num_classes, batch_size=32)
    print(f"[SETUP] Inference device: {api.device}", flush=True)

    # 5. Pre-compute victim scores on eval set for the confidence-gap diagnostic
    print(
        "\n[SETUP] Pre-computing victim confidence scores on eval set …",
        flush=True,
    )
    t0 = time.time()
    eval_member_scores    = api.predict(np.array(eval_member_paths,    dtype=object))
    eval_nonmember_scores = api.predict(np.array(eval_nonmember_paths, dtype=object))
    print(
        f"  {len(eval_member_paths) + len(eval_nonmember_paths)} eval images "
        f"queried in {time.time() - t0:.1f}s",
        flush=True,
    )

    # ── Confidence-gap diagnostic (uses ALL eval samples, not just first 50) ──
    member_max_conf    = np.max(eval_member_scores,    axis=1).mean()
    nonmember_max_conf = np.max(eval_nonmember_scores, axis=1).mean()

    print(f"\n[DIAG] Mean max-confidence on members:     {member_max_conf:.4f}")
    print(f"[DIAG] Mean max-confidence on non-members: {nonmember_max_conf:.4f}")
    print(
        f"[DIAG] Confidence gap (member - nonmember): "
        f"{member_max_conf - nonmember_max_conf:+.4f}  "
        f"(positive = MIA signal exists)"
    )
    sys.stdout.flush()

    if member_max_conf - nonmember_max_conf < 0.05:
        print(
            "\n  WARNING: Confidence gap is very small. "
            "The victim model may not be sufficiently overfitted. "
            "Check memorization_gap in victim_meta.json.",
            flush=True,
        )

    # 6. Run both attacks
    total_start = time.time()

    baseline_result = run_baseline_attack(
        api, pool_all, eval_member_paths, eval_nonmember_paths, num_classes
    )
    variance_result = run_variance_attack(
        api, pool_all, eval_member_paths, eval_nonmember_paths, num_classes
    )

    total_time = time.time() - total_start

    # 7. Final comparison table
    print_banner("FINAL COMPARISON: BASELINE vs VARIANCE-ENHANCED MIA")

    all_results = [baseline_result, variance_result]
    print(
        f"\n  {'Attack Method':<45s}  {'Acc':>7s}  {'Prec':>7s}  "
        f"{'Rec':>7s}  {'F1':>7s}"
    )
    print("  " + "-" * 75)
    for r in all_results:
        print(
            f"  {r['Model']:<45s}  {r['Accuracy']:7.4f}  "
            f"{r['Precision']:7.4f}  {r['Recall']:7.4f}  {r['F1']:7.4f}"
        )

    print(f"\n  Random baseline:      0.5000")
    print(f"  Total attack runtime: {total_time:.1f}s")
    sys.stdout.flush()

    # 8. Save results to text file
    with open(RESULTS_PATH, "w") as f:
        f.write("NIH Chest X-ray — Shadow Model MIA Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Victim architecture:  {meta.get('architecture', 'unknown')}\n")
        f.write(f"Num classes:          {num_classes}\n")
        f.write(f"Label names:          {meta.get('label_names', [])}\n")
        f.write(f"Victim train_acc:     {meta.get('final_train_acc', 0):.4f}\n")
        f.write(f"Victim val_acc:       {meta.get('final_val_acc', 0):.4f}\n")
        f.write(
            f"Memorization gap:     "
            f"{meta.get('memorization_gap', 0) * 100:+.2f}%\n"
        )
        f.write(f"Confidence gap:       {member_max_conf - nonmember_max_conf:+.4f}\n")
        f.write(f"Shadow models:        {NUM_SHADOW_MODELS}\n")
        f.write(f"Shadow dataset size:  {SHADOW_DATASET_SIZE} / model\n")
        f.write(f"Pool size:            {len(pool_all)}\n")
        f.write(f"Eval size:            {len(eval_member_paths) + len(eval_nonmember_paths)}\n\n")

        f.write(
            f"{'Attack Method':<45s}  {'Acc':>7s}  {'Prec':>7s}  "
            f"{'Rec':>7s}  {'F1':>7s}\n"
        )
        f.write("-" * 75 + "\n")
        for r in all_results:
            f.write(
                f"{r['Model']:<45s}  {r['Accuracy']:7.4f}  "
                f"{r['Precision']:7.4f}  {r['Recall']:7.4f}  {r['F1']:7.4f}\n"
            )
        f.write(f"\nRandom baseline: 0.5000\n")
        f.write(f"Total runtime:   {total_time:.1f}s\n")

    print(f"\n  Results saved to: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
