"""
mia_variance.py
===============
Variance-Enhanced Membership Inference Attack (VarianceMIA)

Extends the standard MIA by adding one extra feature per sample:
    variance_of_max — the variance of the MAX sigmoid score across all
    shadow models when they are queried on that image.

Intuition
---------
  • Members of the victim model tend to produce CONSISTENTLY HIGH confidence
    scores across shadow models (low variance).
  • Non-members produce more VARIABLE confidence scores (higher variance).
  • By appending this variance signal to the 15 victim confidence scores,
    the attack model receives 16 features instead of 15, improving accuracy.

Attack dataset columns
----------------------
    class_0, …, class_14,   ← victim sigmoid scores (from shadow model)
    variance_of_max,         ← var of max(shadow_i(x)) across all shadow models
    is_part_of_dataset       ← 1 = member, 0 = non-member

At attack time
--------------
    features = [victim_scores (15-dim), variance_of_max (1-dim)]  = 16-dim
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mia import MIA, ModelParameters, API


class VarianceMIA(MIA):
    """Variance-Enhanced Membership Inference Attack.

    Inherits all parameters and constructor arguments from MIA.
    Overrides:
      - ``_prepare_attack_dataset()`` → adds variance_of_max feature
      - ``_train_attack_model()``     → trains on 16-dim features
      - ``attack()``                  → computes variance at inference time
      - ``evaluate()``                → uses extended attack pipeline
    """

    # ------------------------------------------------------------------
    # Helper: padded predict_proba from one shadow model
    # ------------------------------------------------------------------
    def _padded_predict_proba(self, model, data: np.ndarray) -> np.ndarray:
        """Return (N, num_classes) sigmoid proba; handles partial class sets."""
        raw = model.predict_proba(data)

        if (
            hasattr(model, "architecture")
            or (isinstance(raw, np.ndarray) and raw.ndim == 2
                and raw.shape[1] == self.num_classes)
        ):
            return raw

        if raw.shape[1] == self.num_classes:
            return raw

        full = np.zeros((raw.shape[0], self.num_classes))
        for col_i, cls_label in enumerate(model.classes_):
            if int(cls_label) < self.num_classes:
                full[:, int(cls_label)] = raw[:, col_i]
        return full

    # ------------------------------------------------------------------
    # Helper: compute variance of max-confidence across all shadow models
    # ------------------------------------------------------------------
    def _compute_cross_model_variance(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query all shadow models on `data` and compute variance of max score.

        Parameters
        ----------
        data : np.ndarray of str, shape (N,)
            Image file paths.

        Returns
        -------
        variance : np.ndarray, shape (N, 1)
            Per-sample variance of max(sigmoid) across shadow models.
        stacked_max : np.ndarray, shape (num_shadow_models, N)
            Raw max scores from each shadow model (for debugging).
        """
        all_max = []
        for idx, info in self.shadow_models.items():
            model  = info["model"]
            scores = self._padded_predict_proba(model, data)      # (N, C)
            max_s  = np.max(scores, axis=1)                        # (N,)
            all_max.append(max_s)

        stacked_max = np.stack(all_max, axis=0)                    # (M, N)
        variance    = np.var(stacked_max, axis=0).reshape(-1, 1)   # (N, 1)
        return variance, stacked_max

    # ------------------------------------------------------------------
    # Step 2: Build attack dataset (with variance feature)
    # ------------------------------------------------------------------
    def _prepare_attack_dataset(self):
        """Build self.attack_dataset with 16 features (confidence + variance)."""
        n           = len(self.unlabelled_data)
        all_indices = set(range(n))

        if self.attack_model_dataset_size is not None:
            num_per_model = self.attack_model_dataset_size // (
                2 * self.num_shadow_models
            )
        else:
            num_per_model = self.shadow_model_dataset_size // 2

        rng  = np.random.RandomState(self.random_state + 1)
        rows = []

        for idx, info in self.shadow_models.items():
            model         = info["model"]
            train_indices = info["train_indices"]
            non_train_idx = all_indices - train_indices

            # --- Positive samples (member=1) ---
            pos_pool = list(train_indices)
            k_pos    = min(num_per_model, len(pos_pool))
            pos_idx  = rng.choice(pos_pool, size=k_pos, replace=False)
            pos_data = self.unlabelled_data[pos_idx]

            pos_conf = self._padded_predict_proba(model, pos_data)   # (k, C)
            pos_var, _ = self._compute_cross_model_variance(pos_data) # (k, 1)

            for j in range(k_pos):
                row = {f"class_{c}": pos_conf[j, c] for c in range(self.num_classes)}
                row["variance_of_max"]    = pos_var[j, 0]
                row["is_part_of_dataset"] = 1
                rows.append(row)

            # --- Negative samples (non-member=0) ---
            neg_pool = list(non_train_idx)
            k_neg    = min(num_per_model, len(neg_pool))
            neg_idx  = rng.choice(neg_pool, size=k_neg, replace=False)
            neg_data = self.unlabelled_data[neg_idx]

            neg_conf = self._padded_predict_proba(model, neg_data)
            neg_var, _ = self._compute_cross_model_variance(neg_data)

            for j in range(k_neg):
                row = {f"class_{c}": neg_conf[j, c] for c in range(self.num_classes)}
                row["variance_of_max"]    = neg_var[j, 0]
                row["is_part_of_dataset"] = 0
                rows.append(row)

        self.attack_dataset = pd.DataFrame(rows)
        n_pos = int(self.attack_dataset["is_part_of_dataset"].sum())
        n_neg = len(self.attack_dataset) - n_pos
        print(
            f"  Attack dataset built: {len(self.attack_dataset)} samples "
            f"({n_pos} positive, {n_neg} negative), "
            f"{self.num_classes + 1} features (confidence + variance).",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Step 3: Train attack model on 16-dim features
    # ------------------------------------------------------------------
    def _train_attack_model(self):
        conf_cols    = [f"class_{c}" for c in range(self.num_classes)]
        feature_cols = conf_cols + ["variance_of_max"]

        X = self.attack_dataset[feature_cols].values
        y = self.attack_dataset["is_part_of_dataset"].values

        self.attack_model = self.attack_model_parameters.build(
            random_state=self.random_state
        )
        self.attack_model.fit(X, y)
        print(
            f"  Attack model trained ({self.attack_model_parameters.model_type}) "
            f"on {X.shape[1]} features.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Attack: victim confidence + cross-shadow variance → membership
    # ------------------------------------------------------------------
    def attack(self, data: np.ndarray, return_confidence: bool = False):
        """Predict membership using 16-dim features.

        Parameters
        ----------
        data : np.ndarray of str, shape (N,)
            Image file paths.
        return_confidence : bool
            If True, also return attack model probabilities.

        Returns
        -------
        predictions : np.ndarray of {0, 1}
        confidences : np.ndarray (only if return_confidence=True)
        """
        if not self._is_trained:
            raise RuntimeError("Call .execute() before .attack().")

        # Victim confidence scores (N, num_classes)
        victim_scores = np.asarray(self.victim_model_api.predict(data))
        if victim_scores.ndim == 1:
            victim_scores = victim_scores.reshape(1, -1)

        # Cross-shadow variance (N, 1)
        variance, _ = self._compute_cross_model_variance(data)

        # Combine into 16-dim feature vector
        features = np.hstack([victim_scores, variance])

        predictions = self.attack_model.predict(features)

        if return_confidence:
            if hasattr(self.attack_model, "predict_proba"):
                proba = self.attack_model.predict_proba(features)
            else:
                proba = self.attack_model.decision_function(features)
            return predictions, proba

        return predictions

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def evaluate(
        self,
        member_data: np.ndarray,
        non_member_data: np.ndarray,
    ) -> dict:
        """Evaluate attack on known member / non-member image paths."""
        X      = np.concatenate([member_data, non_member_data])
        y_true = np.concatenate([
            np.ones(len(member_data)),
            np.zeros(len(non_member_data)),
        ])
        y_pred = self.attack(X)
        return {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
        }

    # ------------------------------------------------------------------
    def __repr__(self):
        status = "trained" if self._is_trained else "not trained"
        return (
            f"VarianceMIA(num_classes={self.num_classes}, "
            f"num_shadow_models={self.num_shadow_models}, "
            f"shadow={self.shadow_model_parameters!r}, "
            f"attack={self.attack_model_parameters!r}, "
            f"features=confidence+variance_of_max, "
            f"status={status})"
        )
