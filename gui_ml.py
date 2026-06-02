"""ML engine mixin for SpectrometerApp.

Provides: spectral preprocessing, training worker, hierarchical
classification, confidence estimation, batch preprocessing,
stratified memory management.
"""

import os, math, json, csv, collections, time, sqlite3
import tkinter as tk
from tkinter import messagebox
import numpy as np
import scipy.signal
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import joblib


class MLEngineMixin:
    """ML training, preprocessing, and inference mixed into SpectrometerApp."""

    # Confidence threshold for prediction acceptance.
    # Predictions with confidence below this value are rejected as UNKNOWN.
    CONFIDENCE_THRESHOLD = 60.0

    # ── File path resolution ──────────────────────────────────────────
    def _get_task_data_paths(self, task):
        """Return dict of all data file paths for a task using sanitized name.

        Checks for name-based files first; falls back to ID-based files
        for backward compatibility, migrating old files on first access.
        """
        data_dir = "data"
        # Import TaskManagerMixin's static method via the class hierarchy
        safe_name = self._sanitize_name(task["name"])
        task_id = task.get("id", "unknown")

        # New-style paths (name-based)
        paths = {
            "csv": os.path.join(data_dir, f"task_{safe_name}_dataset.csv"),
            "X_cache": os.path.join(data_dir, f"task_{safe_name}_X.npy"),
            "y_cache": os.path.join(data_dir, f"task_{safe_name}_y.npy"),
            "perf": os.path.join(data_dir, f"task_{safe_name}_perf.jsonl"),
            "inference": os.path.join(data_dir, f"task_{safe_name}_inference.jsonl"),
        }

        # Old-style paths (ID-based) for fallback
        old_paths = {
            "csv": os.path.join(data_dir, f"task_{task_id}_dataset.csv"),
            "X_cache": os.path.join(data_dir, f"task_{task_id}_X.npy"),
            "y_cache": os.path.join(data_dir, f"task_{task_id}_y.npy"),
            "perf": os.path.join(data_dir, f"task_{task_id}_perf.jsonl"),
        }

        # One-time migration: if old files exist but new ones don't, copy them
        new_csv_exists = os.path.exists(paths["csv"])
        new_X_exists = os.path.exists(paths["X_cache"])
        old_csv_exists = os.path.exists(old_paths["csv"])
        old_X_exists = os.path.exists(old_paths["X_cache"])

        if (old_csv_exists or old_X_exists) and not (new_csv_exists or new_X_exists):
            try:
                import shutil
                for key in paths:
                    if os.path.exists(old_paths[key]):
                        shutil.copy2(old_paths[key], paths[key])
                        print(f"[MIGRATE] {old_paths[key]} -> {paths[key]}")
            except Exception as e:
                print(f"[MIGRATE] Migration failed, using old paths: {e}")
                return old_paths

        return paths

    # ── Spectral preprocessing ───────────────────────────────────────
    def _preprocess_spectrum(self, absorbances, algo_name, scaler=None, config=None):
        if absorbances is None or len(absorbances) < 2:
            return None, None

        if config is None:
            config = {}
        snv_enabled = config.get("snv", True)
        msc_enabled = config.get("msc", False)
        baseline_enabled = config.get("baseline", False)
        deriv_order = config.get("deriv", None)
        sg_window = config.get("sg_window", 11)
        sg_poly = config.get("sg_poly", 3)
        scaling_enabled = config.get("scaling", None)
        norm_type = config.get("norm", None)

        if sg_window % 2 == 0:
            sg_window += 1
        if sg_window <= sg_poly:
            sg_window = sg_poly + 2
            if sg_window % 2 == 0:
                sg_window += 1

        need_scaler_default = algo_name in (
            "SVM (Support Vector)", "KNN (K-Nearest)",
            "PCR (Principal Comp)", "SVR (Support Vector)",
            "LDA (Linear Discriminant)"
        )
        need_deriv_default = algo_name in (
            "SVM (Support Vector)", "KNN (K-Nearest)",
            "PCR (Principal Comp)", "SVR (Support Vector)",
            "LDA (Linear Discriminant)"
        )

        if scaling_enabled is None:
            scaling_enabled = need_scaler_default
        if deriv_order is None:
            deriv_order = 1 if need_deriv_default else 0

        X = np.array(absorbances, dtype=np.float64).reshape(1, -1)

        if snv_enabled:
            X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)

        if baseline_enabled:
            X = X - self._baseline_als(X.flatten())

        if msc_enabled and X.shape[0] > 1:
            X = self._msc_correct(X)
        elif msc_enabled:
            pass

        X_filtered = scipy.signal.savgol_filter(
            X, window_length=sg_window, polyorder=sg_poly,
            deriv=deriv_order, axis=1
        )

        if norm_type == "l2":
            norms = np.linalg.norm(X_filtered, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            X_filtered = X_filtered / norms

        if scaling_enabled:
            if X_filtered.shape[0] == 1:
                return X_filtered, None
            if scaler is not None:
                X_scaled = scaler.transform(X_filtered)
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_filtered)
            return X_scaled, scaler
        else:
            return X_filtered, None

    # ── Confidence estimation ────────────────────────────────────────
    def _compute_confidence(self, model, X):
        try:
            # SVC: decision_function hyperplane distance + sigmoid
            if isinstance(model, SVC):
                d = model.decision_function(X)
                if d.ndim == 1:
                    mean_abs_d = abs(float(d[0]))
                else:
                    mean_abs_d = np.mean(np.abs(d))
                confidence = 100.0 / (1.0 + math.exp(-mean_abs_d))
                return confidence, "decision"

            # RF: vote ratio
            if isinstance(model, RandomForestClassifier):
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)
                    n_trees = getattr(model, 'n_estimators', '?')
                    return np.max(probs) * 100, f"vote, {n_trees} trees"
                return 0, "--"

            # KNN: neighbor vote ratio
            if isinstance(model, KNeighborsClassifier):
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)
                    k = getattr(model, 'n_neighbors', '?')
                    return np.max(probs) * 100, f"vote, {k}-NN"
                return 0, "--"

            # Fallback
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                return np.max(probs) * 100, "proba"
            return 0, "--"
        except Exception:
            return 0, "--"

    # ── Prediction metrics logging ────────────────────────────────────
    def _log_prediction_metrics(self, task, prediction, confidence, source,
                                 accepted, mode="single", batch_size=1,
                                 hierarchical=False, vote_counts=None):
        """Write one JSON line to the task's inference log file."""
        paths = self._get_task_data_paths(task)
        log_path = paths["inference"]
        entry = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "type": "inference",
            "mode": mode,
            "prediction": str(prediction),
            "accepted": bool(accepted),
            "confidence": float(round(float(confidence), 2)),
            "confidence_source": str(source),
            "threshold": float(self.CONFIDENCE_THRESHOLD),
            "hierarchical": bool(hierarchical) if hierarchical is not None else False,
            "batch_size": int(batch_size),
            "task_name": str(task.get("name", "")),
        }
        if vote_counts is not None:
            entry["vote_counts"] = vote_counts
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Performance data loading ──────────────────────────────────────
    def _load_latest_perf(self, task):
        """Read the latest performance entry from the task's perf.jsonl file.

        Returns a dict with keys: timestamp, n_samples, accuracy,
        confusion_matrix (2D list), class_distribution, or None.
        """
        paths = self._get_task_data_paths(task)
        perf_path = paths["perf"]
        if not os.path.exists(perf_path):
            return None
        try:
            with open(perf_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if not lines:
                return None
            # Last non-empty line
            for line in reversed(lines):
                stripped = line.strip()
                if stripped:
                    return json.loads(stripped)
            return None
        except (json.JSONDecodeError, IOError, IndexError):
            return None

    def _format_perf_text(self, perf_entry):
        """Format a performance entry into a fixed-width display string.

        Produces:
            MODEL ACCURACY: 93.3%  (15 samples)

            -- CONFUSION MATRIX --
                        Cotton  Nylon  Polyester
               Cotton      5      0        0
                Nylon      0      4        1
            Polyester      0      0        5

        Returns an empty string if perf_entry is None.
        """
        if not perf_entry:
            return ""
        acc = perf_entry.get("accuracy", 0)
        n_samples = perf_entry.get("n_samples", 0)
        cm = perf_entry.get("confusion_matrix", [])
        class_dist = perf_entry.get("class_distribution", {})

        lines = [f"MODEL ACCURACY: {acc*100:.1f}%  ({n_samples} samples)"]

        if cm and class_dist:
            classes = sorted(class_dist.keys())
            if len(classes) >= 2:
                lines.append("")
                lines.append("-- CONFUSION MATRIX --")
                max_name = max(len(c) for c in classes)
                col_w = max(max_name, 6) + 2
                # Header row
                header = " " * (max_name + 2)
                for c in classes:
                    header += f"{c:>{col_w}}"
                lines.append(header)
                # Data rows
                for i, actual in enumerate(classes):
                    row = f"  {actual:>{max_name}}"
                    for j in range(len(classes)):
                        val = cm[i][j] if i < len(cm) and j < len(cm[i]) else "-"
                        row += f"{val:>{col_w}}"
                    lines.append(row)
        return "\n".join(lines)

    # ── Baseline / MSC helpers ───────────────────────────────────────
    def _baseline_als(self, y, lam=1e5, p=0.01, n_iter=10):
        y = np.asarray(y, dtype=np.float64).ravel()
        L = len(y)
        D = np.eye(L, dtype=np.float64)
        diag = np.ones(L - 2, dtype=np.float64)
        D += np.diag(-2.0 * np.ones(L - 1, dtype=np.float64), k=-1)
        D += np.diag(diag, k=-2)
        D += np.diag(-2.0 * np.ones(L - 1, dtype=np.float64), k=1)
        D += np.diag(diag, k=2)
        D = lam * (D.T @ D)
        w = np.ones(L, dtype=np.float64)
        for _ in range(n_iter):
            W = np.diag(w)
            Z = np.linalg.solve(W + D, w * y)
            w = p * (y > Z) + (1 - p) * (y < Z)
        return Z

    def _msc_correct(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] <= 1:
            return X
        ref = np.mean(X, axis=0)
        X_msc = np.zeros_like(X)
        for i in range(X.shape[0]):
            A = np.vstack([ref, np.ones_like(ref)]).T
            coef = np.linalg.lstsq(A, X[i], rcond=None)[0]
            X_msc[i] = (X[i] - coef[1]) / coef[0]
        return X_msc

    # ── Hierarchical classification ──────────────────────────────────
    def _hierarchical_train(self, X, y, base_cls):
        means = np.mean(X, axis=1)
        median_mean = np.median(means)
        y_level1 = np.where(means >= median_mean, "HIGH", "LOW")
        unique_levels = np.unique(y_level1)
        if len(unique_levels) < 2:
            clf1 = None
            low_idx = np.arange(len(y))
        else:
            clf1 = base_cls.__class__(**base_cls.get_params()) if hasattr(base_cls, 'get_params') else base_cls.__class__()
            clf1.fit(X, y_level1)
            low_idx = np.where(y_level1 == "LOW")[0]
        if len(low_idx) < 2:
            clf2 = None
            low_labels = None
        else:
            y_low = y[low_idx]
            low_labels = np.unique(y_low)
            if len(low_labels) < 2:
                clf2 = None
            else:
                X_low = X[low_idx]
                clf2 = base_cls.__class__(**base_cls.get_params()) if hasattr(base_cls, 'get_params') else base_cls.__class__()
                clf2.fit(X_low, y_low)
        self._hier_level_map = {
            "clf1": clf1, "clf2": clf2,
            "median_mean": median_mean,
            "low_labels": low_labels
        }
        return clf1, clf2

    def _hierarchical_predict(self, X):
        mm = self._hier_level_map
        clf1 = mm["clf1"]
        clf2 = mm["clf2"]
        means = np.mean(X, axis=1)
        if clf1 is not None:
            level_pred = clf1.predict(X)
        else:
            level_pred = np.where(means >= mm["median_mean"], "HIGH", "LOW")
        results = np.empty(len(X), dtype=object)
        high_mask = level_pred == "HIGH"
        low_mask = level_pred == "LOW"
        if np.any(high_mask):
            X_high = X[high_mask]
            means_high = means[high_mask]
            high_classes = []
            for i in range(len(X_high)):
                dist_to_high = abs(means_high[i] - mm["median_mean"])
                if clf2 is not None and mm["low_labels"] is not None:
                    high_classes.append(mm["low_labels"][0])
                else:
                    high_classes.append("UNKNOWN")
            results[high_mask] = high_classes
        if np.any(low_mask) and clf2 is not None and mm["low_labels"] is not None and len(mm["low_labels"]) >= 2:
            results[low_mask] = clf2.predict(X[low_mask])
        elif np.any(low_mask):
            if clf2 is not None and mm["low_labels"] is not None:
                results[low_mask] = mm["low_labels"][0]
            else:
                results[low_mask] = "UNKNOWN"
        return results

    # ── Batch preprocessing ──────────────────────────────────────────
    def _preprocess_batch(self, X_raw, algo_name):
        """Vectorized batch preprocessing: process all historical rows at once."""
        if X_raw is None or len(X_raw) < 1:
            return None
        results = []
        for i in range(len(X_raw)):
            row = X_raw[i]
            row_proc, _ = self._preprocess_spectrum(np.array(row), algo_name)
            if row_proc is not None:
                results.append(row_proc)
        return np.vstack(results) if results else None

    # ── Stratified memory management ─────────────────────────────────
    def _get_stratified_keep_indices(self, df, max_total, log_fn=None):
        """Return indices of rows to keep (not the trimmed dataframe), for cache alignment."""
        labels = df.iloc[:, 1].values
        unique_labels, counts = np.unique(labels, return_counts=True)
        n_classes = len(unique_labels)
        min_per_class = max(3, max_total // n_classes // 2)
        keep_indices = []
        for label in unique_labels:
            label_idx = np.where(labels == label)[0]
            target = min(max(len(label_idx), min_per_class), max(2 * max_total // n_classes, min_per_class))
            if len(label_idx) <= target:
                keep_indices.extend(label_idx.tolist())
            else:
                keep_indices.extend(label_idx[-target:].tolist())
        keep_indices.sort()
        total_kept = len(keep_indices)
        while total_kept > max_total:
            label_counts = {l: 0 for l in unique_labels}
            for idx in keep_indices:
                label_counts[labels[idx]] += 1
            most_label = max(label_counts, key=label_counts.get)
            for i in range(len(keep_indices)):
                if labels[keep_indices[i]] == most_label:
                    del keep_indices[i]
                    total_kept -= 1
                    break
        keep_indices.sort()
        if log_fn:
            keep_counts = collections.Counter(labels[keep_indices])
            log_fn(f"[MEM_OPT] Stratified window: kept {len(keep_indices)}/{len(df)} samples, per-class: {dict(keep_counts)}")
        return keep_indices

    def _stratified_trim_df(self, df, max_total, log_fn=None):
        labels = df.iloc[:, 1].values
        unique_labels, counts = np.unique(labels, return_counts=True)
        n_classes = len(unique_labels)
        min_per_class = max(3, max_total // n_classes // 2)
        keep_indices = []
        for label in unique_labels:
            label_idx = np.where(labels == label)[0]
            target = min(max(len(label_idx), min_per_class), max(2 * max_total // n_classes, min_per_class))
            if len(label_idx) <= target:
                keep_indices.extend(label_idx.tolist())
            else:
                keep_indices.extend(label_idx[-target:].tolist())
        keep_indices.sort()
        total_kept = len(keep_indices)
        while total_kept > max_total:
            label_counts = {l: 0 for l in unique_labels}
            for idx in keep_indices:
                label_counts[labels[idx]] += 1
            most_label = max(label_counts, key=label_counts.get)
            for i in range(len(keep_indices)):
                if labels[keep_indices[i]] == most_label:
                    del keep_indices[i]
                    total_kept -= 1
                    break
        keep_indices.sort()
        trimmed = df.iloc[keep_indices].copy()
        if log_fn:
            keep_counts = collections.Counter(trimmed.iloc[:, 1].values)
            log_fn(f"[MEM_OPT] Stratified window: kept {len(trimmed)}/{len(df)} samples, per-class: {dict(keep_counts)}")
        return trimmed

    # ── Training worker ──────────────────────────────────────────────
    def _train_model_worker(self, batch_absorbances=None):
        """搭载在线学习（Online Learning）引擎的训练逻辑，支持滑动窗口与热启动微调。

        Args:
            batch_absorbances: Optional list of absorbance arrays for batch mode.
                               When None, reads single spectrum from self.scan_data.
        """
        if getattr(self, 'is_training_active', False): return
        self.is_training_active = True

        # Check if we're in batch mode
        is_batch = batch_absorbances is not None and len(batch_absorbances) > 0

        def log(text):
            self.root.after(0, lambda: self.train_console.insert(tk.END, f"> {text}"))
            self.root.after(0, lambda: self.train_console.yview(tk.END))

        try:
            current_model = None
            can_train = False

            task = self.tasks[self.current_task_idx]
            task_type = task.get("task_type", "Classification")
            model_path = task.get("model_path")
            task_id = task.get("id", "unknown")

            # ── Resolve data file paths (name-based with backward compat) ──
            paths = self._get_task_data_paths(task)
            csv_filename = paths["csv"]
            cache_X_path = paths["X_cache"]
            cache_y_path = paths["y_cache"]
            data_dir = os.path.dirname(csv_filename) or "data"

            if is_batch:
                log(f"Batch training: processing {len(batch_absorbances)} spectra...")
            else:
                log("Optics scan successfully captured via central pipeline.")

            # ── Load/reject current sample(s) ──
            algorithm = task.get("algorithm", "")
            X_batch_list = []
            raw_absorbances_list = []  # for CSV save

            if is_batch:
                for i, abs_data in enumerate(batch_absorbances):
                    X_proc, _ = self._preprocess_spectrum(np.array(abs_data), algorithm)
                    if X_proc is not None:
                        X_batch_list.append(X_proc)
                        raw_absorbances_list.append(abs_data)
                    else:
                        log(f"[WARN] Spectrum {i+1}/{len(batch_absorbances)} preprocessing failed, skipping")
                if not X_batch_list:
                    log("[ERROR] All batch spectra failed preprocessing!")
                    return
                X_filtered = X_batch_list[0]  # reference for feature dim
                scaler = None
            else:
                raw_absorbances = self.scan_data.get('absorbance', [])
                if not raw_absorbances:
                    log("[WARN] No valid absorbance data found!")
                    return
                absorbances = raw_absorbances

                raw_intensities = self.scan_data.get('intensity', [])
                if raw_intensities and np.ptp(raw_intensities) < 1e-6:
                    log(f"[REJECT] Dead sensor: all {len(raw_intensities)} intensity values = {raw_intensities[0]}")
                    self.root.after(0, lambda: messagebox.showwarning(
                        "Bad Scan", "Scan produced constant values.\nPlease check sensor and retry."
                    ))
                    self.root.after(0, lambda: self.start_train_btn.config(state=tk.NORMAL))
                    return

                log("Executing algorithm-specific spectral preprocessing...")
                X_filtered, scaler = self._preprocess_spectrum(np.array(absorbances), algorithm)
                if X_filtered is None:
                    log("[ERROR] Preprocessing failed: invalid spectral data!")
                    return
                X_batch_list = [X_filtered]
                raw_absorbances_list = [absorbances]

            if task_type == "Classification":
                current_y = self.train_tag_var.get().strip()
                if not current_y or current_y == "Error: No Tags Found":
                    log("[ERROR] 无效标签，请先选择样本类别！")
                    return
            else:
                try: current_y = float(self.train_target_var.get())
                except ValueError:
                    log("[ERROR] 回归任务目标值无效！")
                    return

            X_all, y_all = None, None
            MAX_MEMORY_SAMPLES = 500  # 限制最大样本量，旧数据将被自动"遗忘"，保证树莓派永远不卡死

            if os.path.exists(csv_filename) and os.path.exists(cache_X_path):
                try:
                    # Fast path: load cached preprocessed features (no re-processing of history)
                    X_hist = np.load(cache_X_path)
                    y_hist = np.load(cache_y_path, allow_pickle=True)
                    log(f"[CACHE] Loaded {len(y_hist)} preprocessed samples from feature cache")

                    # Apply sliding window trim to cached data
                    if len(y_hist) > MAX_MEMORY_SAMPLES:
                        # Build temporary dataframe for stratified trim, then subset cache
                        df_tmp = pd.read_csv(csv_filename)
                        keep_idx = self._get_stratified_keep_indices(df_tmp, MAX_MEMORY_SAMPLES, log)
                        if keep_idx is not None and len(keep_idx) < len(y_hist):
                            X_hist = X_hist[keep_idx]
                            y_hist = y_hist[keep_idx]
                            log(f"[CACHE] Trimmed to {len(y_hist)} samples via stratified window")

                    # Merge batch (X_batch: N rows, y_batch: N labels)
                    X_batch = np.vstack(X_batch_list)
                    y_batch = np.array([current_y] * len(X_batch_list))
                    X_all = np.vstack([X_hist, X_batch]) if len(X_hist) > 0 else X_batch
                    y_all = np.append(y_hist, y_batch) if len(y_hist) > 0 else y_batch
                except Exception as e:
                    log(f"[CACHE] Stale or corrupt, falling back to CSV: {e}")
                    os.path.exists(cache_X_path) and os.remove(cache_X_path)
                    os.path.exists(cache_y_path) and os.remove(cache_y_path)
                    X_all, y_all = None, None

            if X_all is None and os.path.exists(csv_filename):
                # Slow path (first run or cache miss): re-process all historical samples from CSV
                try:
                    df = pd.read_csv(csv_filename)
                    if not df.empty:
                        if len(df) > MAX_MEMORY_SAMPLES:
                            df = self._stratified_trim_df(df, MAX_MEMORY_SAMPLES, log)

                        X_hist_raw = df.iloc[:, 2:].values
                        y_hist = df.iloc[:, 1].values

                        # Batch-preprocess all rows at once (vectorized, much faster than per-row)
                        X_hist_processed = self._preprocess_batch(X_hist_raw, algorithm)

                        if X_hist_processed is not None and len(X_hist_processed) > 0:
                            X_batch = np.vstack(X_batch_list)
                            y_batch = np.array([current_y] * len(X_batch_list))
                            X_all = np.vstack([X_hist_processed, X_batch])
                            y_all = np.append(y_hist, y_batch)
                            # Build cache for next time
                            np.save(cache_X_path, np.vstack([X_hist_processed, X_batch]))
                            np.save(cache_y_path, np.append(y_hist, y_batch))
                            log(f"[CACHE] Built feature cache with {len(y_all)} samples")
                except Exception as e:
                    log(f"[WARN] Read historical CSV failed: {e}")

            if X_all is None or y_all is None:
                X_batch = np.vstack(X_batch_list)
                y_batch = np.array([current_y] * len(X_batch_list))
                X_all = X_batch
                y_all = y_batch

            if task_type == "Classification" and X_all is not None and y_all is not None and len(y_all) >= 2:
                # Use last sample of batch for outlier detection (representative)
                X_batch_arr = np.vstack(X_batch_list)
                new_x = X_batch_arr[-1:] if len(X_batch_list) > 1 else X_batch_arr
                # Mask: historical samples (all except the current batch) that match this class
                n_batch = len(X_batch_list)
                historical_mask = np.zeros(len(y_all), dtype=bool)
                if len(y_all) > n_batch:
                    historical_mask[:len(y_all)-n_batch] = True
                same_class_mask = historical_mask & (y_all == current_y)
                if np.sum(same_class_mask) >= 3:
                    from sklearn.metrics.pairwise import euclidean_distances
                    X_same_class = X_all[same_class_mask]
                    dists = euclidean_distances(new_x, X_same_class).flatten()
                    min_dist = np.min(dists)
                    if min_dist < 0.001:
                        log(f"[NEAR-DUP] New sample extremely close to existing (dist={min_dist:.6f})")
                if np.sum(same_class_mask) >= 5:
                    from sklearn.covariance import EllipticEnvelope
                    X_same_class = X_all[same_class_mask]
                    try:
                        ee = EllipticEnvelope(contamination=0.1, random_state=0)
                        ee.fit(X_same_class)
                        if ee.predict(new_x)[0] == -1:
                            log("[OUTLIER] New sample flagged as outlier vs class history")
                    except Exception:
                        pass

            can_train = True
            if task_type == "Classification":
                unique_classes = np.unique(y_all)
                if len(unique_classes) < 2:
                    log(f"[WAITING] 采集样本数: {len(y_all)}. 分类需至少2种标签才能启动AI。")
                    can_train = False
            else:
                if len(y_all) < 3:
                    log(f"[WAITING] 采集样本数: {len(y_all)}. 回归需至少3个点才能启动AI。")
                    can_train = False

            if os.path.exists(model_path):
                try:
                    loaded_obj = joblib.load(model_path)
                    current_model = loaded_obj["classifier"] if isinstance(loaded_obj, dict) else loaded_obj
                except Exception as e:
                    log(f"[WARN] Failed to load existing model: {e}")
                    current_model = None

            if can_train and current_model is not None:
                log("Configuring model hyperparameters...")

                core_algo = current_model

                n_samples = len(y_all)


                cv_splits = 1
                if task_type == "Classification":
                    _, class_counts = np.unique(y_all, return_counts=True)
                    min_class_count = np.min(class_counts)
                    if min_class_count >= 2:
                        cv_splits = min(3, min_class_count)

                train_fold_size = int(n_samples * (cv_splits - 1) / cv_splits) if cv_splits > 1 else n_samples
                train_fold_size = max(1, train_fold_size)


                if isinstance(core_algo, KNeighborsClassifier):
                    safe_k = max(1, train_fold_size - 1)
                    core_algo.n_neighbors = min(5, safe_k)
                    log(f"[DEBUG] KNN 锁定 n_neighbors: {core_algo.n_neighbors}")

                elif isinstance(core_algo, PLSRegression):
                    n_features_pls = X_all.shape[1]
                    safe_pls_comp = max(1, train_fold_size - 1)
                    core_algo.n_components = min(10, n_features_pls - 1, safe_pls_comp)
                    log(f"[DEBUG] PLSR 锁定 n_components: {core_algo.n_components}")

                # 3. 随机森林 适配 (死锁预警：极小样本下建树过多会导致内存与性能浪费)
                # 3. 随机森林 适配 (解除维度封印)
                elif isinstance(core_algo, RandomForestClassifier):
                    from sklearn.utils.class_weight import compute_class_weight

                    # 🔥 核心修复 1：彻底关闭 warm_start。
                    # 让每次训练都根据当前的 PCA 维度瞬间重建森林，永不崩溃！
                    core_algo.warm_start = False

                    # 小样本建50棵树提高投票粒度，数据够了建100棵树保证精度
                    if n_samples < 10:
                        core_algo.n_estimators = 50
                    else:
                        core_algo.n_estimators = 100

                    # 依然保留防偏科的权重平衡机制
                    if task_type == "Classification":
                        unique_y = np.unique(y_all)
                        if len(unique_y) > 1:
                            cw = compute_class_weight('balanced', classes=unique_y, y=y_all)
                            core_algo.class_weight = dict(zip(unique_y, cw))

                    log(f"[DEBUG] RF 动态适配: n_estimators={core_algo.n_estimators}")

                    log(f"[DEBUG] RF 在线增量更新: n_estimators={core_algo.n_estimators}")

                elif isinstance(core_algo, SVC):
                    # 【核心】防偏科机制：如果连续扫 A 类，强制平衡权重保护 B 类
                    core_algo.class_weight = "balanced"
                    log("[DEBUG] SVM 在线引擎状态校验: READY, Class Balanced")

                elif isinstance(core_algo, SVR):
                    log("[DEBUG] SVR 在线引擎状态校验: READY")

                elif isinstance(core_algo, LinearDiscriminantAnalysis):
                    log("[DEBUG] LDA 在线引擎状态校验: READY")

                log("Executing REAL Scikit-Learn Fitting...")
                core_algo.fit(X_all, y_all)
                log("Optimization converged perfectly.")
                current_model = core_algo

                # --- Train hierarchical intensity-split classifier for classification tasks ---
                hierarchical_data = None
                if task_type == "Classification":
                    try:
                        log("Training hierarchical intensity-split classifier...")
                        clf1, clf2 = self._hierarchical_train(X_all, y_all, current_model)
                        if clf1 is not None or clf2 is not None:
                            hierarchical_data = {
                                "clf1": clf1,
                                "clf2": clf2,
                                "median_mean": self._hier_level_map.get("median_mean"),
                                "low_labels": self._hier_level_map.get("low_labels")
                            }
                            log(f"[HIERARCHICAL] Two-level split trained successfully (median intensity = {hierarchical_data['median_mean']:.4f})")
                        else:
                            log("[HIERARCHICAL] Skipped — insufficient data for a meaningful two-level split")
                    except Exception as e:
                        log(f"[HIERARCHICAL] Training failed, falling back to flat classifier: {e}")
                        hierarchical_data = None

                if task_type == "Classification":
                    y_pred_all = current_model.predict(X_all)
                    report = classification_report(y_all, y_pred_all, output_dict=True, zero_division=0)
                    cm = confusion_matrix(y_all, y_pred_all)
                    acc = float(report.get("accuracy", 0))
                    perf_entry = {
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "n_samples": len(y_all),
                        "accuracy": acc,
                        "confusion_matrix": cm.tolist(),
                        "class_distribution": dict(collections.Counter(y_all))
                    }
                    perf_dir = "data"
                    if not os.path.exists(perf_dir): os.makedirs(perf_dir)
                    perf_path = paths["perf"]
                    with open(perf_path, 'a', encoding='utf-8') as pf:
                        pf.write(json.dumps(perf_entry) + "\n")
                    history = []
                    if os.path.exists(perf_path):
                        with open(perf_path, 'r', encoding='utf-8') as pf:
                            for line in pf:
                                try: history.append(json.loads(line))
                                except: pass
                    if len(history) >= 3:
                        last_3 = history[-3:]
                        a1, a2, a3 = last_3[0]["accuracy"], last_3[1]["accuracy"], last_3[2]["accuracy"]
                        if a3 < a2 and a2 < a1:
                            log(f"[WARN] Model accuracy declining: {a1*100:.1f}% -> {a2*100:.1f}% -> {a3*100:.1f}%")
                        elif a3 > a2:
                            log(f"[TREND] Accuracy improving: {a2*100:.1f}% -> {a3*100:.1f}% (^)")
                        else:
                            log(f"[TREND] Accuracy stable: ~{a3*100:.1f}% (->)")

                # --- 后续的评估与保存逻辑保持不变 ---
                if task_type == "Classification":
                    if min_class_count >= 2:
                        scores = cross_val_score(current_model, X_all, y_all, cv=cv_splits)
                        eval_msg = f"Cross-Validation ({cv_splits}-Fold): {np.mean(scores)*100:.1f}%"
                    else:
                        eval_msg = f"Training Accuracy: {current_model.score(X_all, y_all)*100:.1f}%"

                    result_msg = f"SUCCESS: Training Complete.\n\n[CLASSIFICATION METRIC]\n{eval_msg}\nTotal Samples: {len(y_all)}\n\nYes=Save+Update  No=Save Only  Cancel=Discard"
                else:
                    y_pred = current_model.predict(X_all)
                    rmse = np.sqrt(mean_squared_error(y_all, y_pred))
                    result_msg = f"SUCCESS: Training Complete.\n\n[REGRESSION METRIC]\nTraining RMSE: {rmse:.4f}\nTotal Samples: {len(y_all)}\n\nYes=Save+Update  No=Save Only  Cancel=Discard"

                if task_type == "Classification":
                    from collections import Counter
                    class_dist = Counter(y_all)
                    log(f"[SAMPLING] Class distribution: {dict(class_dist)}")
                    class_mean = np.mean(list(class_dist.values()))
                    for cls, cnt in class_dist.items():
                        if cnt < class_mean * 0.5:
                            log(f"[ADVICE] Suggest scanning more of [{cls}] (only {cnt} vs avg {class_mean:.1f})")
            else:
                result_msg = f"[ DATA ACQUISITION ]\n\nSample acquired successfully.\nTotal Samples in Database: {len(y_all)}\n\nInsufficient data to trigger AI training.\nYes=Save  No=Discard"

            has_model = can_train and current_model is not None

            def prompt_user_decision():
                if has_model:
                    user_choice = messagebox.askyesnocancel("[ EVALUATION ]", result_msg)
                else:
                    user_choice = messagebox.askyesno("[ EVALUATION ]", result_msg)

                if user_choice is True:
                    try:
                        # Save ALL samples (single or batch) to CSV
                        file_exists = os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0
                        with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                first_abs = raw_absorbances_list[0]
                                header = ["timestamp", "label"] + [f"a_{j}" for j in range(len(first_abs))]
                                writer.writerow(header)
                            for abs_data in raw_absorbances_list:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                row_data = [timestamp, current_y] + list(abs_data)
                                writer.writerow(row_data)
                        n_saved = len(raw_absorbances_list)
                        log(f"{n_saved} sample(s) permanently saved to matrix.")

                        # Update feature cache with ALL batch entries
                        if os.path.exists(cache_X_path) and os.path.exists(cache_y_path):
                            try:
                                X_cache = np.load(cache_X_path)
                                y_cache = np.load(cache_y_path, allow_pickle=True)
                                X_new = np.vstack(X_batch_list)
                                y_new = np.array([current_y] * len(X_batch_list))
                                X_cache = np.vstack([X_cache, X_new]) if X_cache.shape[0] > 0 else X_new
                                y_cache = np.append(y_cache, y_new) if len(y_cache) > 0 else y_new
                                np.save(cache_X_path, X_cache)
                                np.save(cache_y_path, y_cache)
                                log(f"[CACHE] Feature cache updated ({len(y_cache)} samples)")
                            except Exception:
                                pass  # silent — cache is optional, CSV is authoritative

                        try:
                            df_check = pd.read_csv(csv_filename)
                            if len(df_check) > 600:
                                trimmed = self._stratified_trim_df(df_check, MAX_MEMORY_SAMPLES, log)
                                trimmed.to_csv(csv_filename, index=False)
                                log(f"[CSV_TRIM] Reduced from {len(df_check)} to {len(trimmed)} rows")
                                # Rebuild cache after trim to stay in sync
                                if os.path.exists(cache_X_path):
                                    os.remove(cache_X_path)
                                if os.path.exists(cache_y_path):
                                    os.remove(cache_y_path)
                        except Exception:
                            pass

                        if can_train and current_model is not None:
                            dump_data = {"classifier": current_model, "feature_mask": list(range(X_batch_list[0].shape[1]))}
                            if scaler is not None:
                                dump_data["preprocessor_scaler"] = scaler
                            if hierarchical_data is not None:
                                dump_data["hierarchical"] = hierarchical_data
                            joblib.dump(dump_data, model_path)

                            conn = sqlite3.connect(self.db_path)
                            cursor = conn.cursor()
                            cursor.execute("UPDATE tasks SET is_trained = 1 WHERE id = ?", (task_id,))
                            conn.commit()
                            conn.close()
                            task["is_trained"] = 1
                            messagebox.showinfo("Success", f"Model deployed with {len(y_all)} total samples!")
                        else:
                            messagebox.showinfo("Success", f"{n_saved} batch sample(s) saved!")

                    except Exception as ex:
                        error_msg = str(ex)
                        self.root.after(0, lambda msg=error_msg: messagebox.showerror("Storage Error", f"Failed to save: {msg}"))
                elif user_choice is False:
                    try:
                        # Save all samples to CSV without model update
                        file_exists = os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0
                        with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                first_abs = raw_absorbances_list[0]
                                header = ["timestamp", "label"] + [f"a_{j}" for j in range(len(first_abs))]
                                writer.writerow(header)
                            for abs_data in raw_absorbances_list:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                row_data = [timestamp, current_y] + list(abs_data)
                                writer.writerow(row_data)
                        log("Data saved without model update.")
                        # Update cache even when model is not retrained
                        if os.path.exists(cache_X_path) and os.path.exists(cache_y_path):
                            try:
                                X_cache = np.load(cache_X_path)
                                y_cache = np.load(cache_y_path, allow_pickle=True)
                                X_new = np.vstack(X_batch_list)
                                y_new = np.array([current_y] * len(X_batch_list))
                                X_cache = np.vstack([X_cache, X_new]) if X_cache.shape[0] > 0 else X_new
                                y_cache = np.append(y_cache, y_new) if len(y_cache) > 0 else y_new
                                np.save(cache_X_path, X_cache)
                                np.save(cache_y_path, y_cache)
                            except Exception:
                                pass
                        messagebox.showinfo("Saved", f"{len(raw_absorbances_list)} sample(s) saved. Model NOT updated.")
                    except Exception as ex:
                        error_msg = str(ex)
                        self.root.after(0, lambda msg=error_msg: messagebox.showerror("Storage Error", f"Failed to save: {msg}"))
                else:
                    log("User discarded training. Matrix remains untouched.")

                self.start_train_btn.config(state=tk.NORMAL)
                self.show_execute_page()

            self.root.after(0, prompt_user_decision)

        except Exception as e:
            error_msg = str(e)
            print(f"\n[ 🔥 REAL FATAL ERROR 🔥 ]\n{error_msg}\n")
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Fatal Pipeline Failure", msg))

        finally:
            self.is_training_active = False
            self.root.after(0, lambda: self.start_train_btn.config(state=tk.NORMAL))

    def _batch_train_worker(self):
        """Batch training entry point: reads accumulated spectra and delegates
        to _train_model_worker with the batch data.
        """
        batch_absorbances = getattr(self, '_batch_absorbances', [])
        if not batch_absorbances:
            print("[BATCH TRAIN] No batch data to train on")
            return
        self._train_model_worker(batch_absorbances=batch_absorbances)

    # ── Legacy ML model runner ───────────────────────────────────────
    def _run_ml_model(self, task_name, waves, absorbances, intensities):
        if absorbances is None or len(absorbances) == 0: return ">> DATA ERROR: EMPTY SCAN <<"
        if task_name not in self.ml_models: return f">> {task_name}: Awaiting model <<"

        try:
            model_dict = self.ml_models[task_name]
            clf, mask = model_dict["clf"], model_dict["mask"]
            X_raw = np.array(absorbances).reshape(1, -1)
            wl_array = np.array(waves)

            mean = np.mean(X_raw, axis=1, keepdims=True)
            std = np.where(np.std(X_raw, axis=1, keepdims=True) == 0, 1e-8, np.std(X_raw, axis=1, keepdims=True))
            X_sg = scipy.signal.savgol_filter((X_raw - mean) / std, window_length=11, polyorder=3, deriv=1, axis=1)

            expected_features = len(mask)
            current_features = X_sg.shape[1]

            if current_features == expected_features:
                X_processed = X_sg
            else:
                idx_start = np.argmin(abs(wl_array - 920))
                idx_end = min(idx_start + expected_features, current_features)
                idx_start = max(0, idx_end - expected_features)
                X_processed = X_sg[:, idx_start:idx_end]

            X_new_opt = X_processed[:, mask]
            prediction = clf.predict(X_new_opt)[0]
            confidence_str = f"\n\nCONFIDENCE: {np.max(clf.predict_proba(X_new_opt))*100:.1f}%" if hasattr(clf, "predict_proba") else ""

            return f">> PREDICTION: {str(prediction).upper()} <<{confidence_str}"
        except Exception as e:
            return f">> INFERENCE ERROR <<\n\n{str(e)}"
