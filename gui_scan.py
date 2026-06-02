"""Scan workflow mixin for SpectrometerApp.

Provides: hardware scan task, CSV save, data processing, UI result
dispatch (scan / predict / train), cleanup, graph drawing.
"""

import os, math, csv, time, threading, json
import numpy as np
import joblib
from datetime import datetime
from tkinter import messagebox
import tkinter as tk


class ScanWorkflowMixin:
    """Scan operations mixed into SpectrometerApp."""

    # ── Graph drawing ────────────────────────────────────────────────
    def set_graph_mode(self, mode):

        self.graph_mode = mode
        for btn in [self.btn_int, self.btn_abs, self.btn_ref]:
            btn.config(bg="#b5b5b5", fg="#1a1a1a", relief=tk.RAISED)
        if mode == 'intensity': self.btn_int.config(bg="#404040", fg="#39ff14", relief=tk.SUNKEN)
        elif mode == 'absorbance': self.btn_abs.config(bg="#404040", fg="#39ff14", relief=tk.SUNKEN)
        elif mode == 'reflectance': self.btn_ref.config(bg="#404040", fg="#39ff14", relief=tk.SUNKEN)
        self._draw_graph()

    def _draw_graph(self):
        self.scan_canvas.delete("all")
        w, h = self.scan_canvas.winfo_width(), self.scan_canvas.winfo_height()
        if w < 10 or h < 10: return

        for i in range(1, 10):
            x, y = i * (w / 10), i * (h / 10)
            self.scan_canvas.create_line(x, 0, x, h, fill="#0a290a", dash=(2, 2))
            self.scan_canvas.create_line(0, y, w, y, fill="#0a290a", dash=(2, 2))

        with self.scan_data_lock:
            waves = self.scan_data['wave']
            y_data = self.scan_data[self.graph_mode]

        if not waves or not y_data or len(waves) != len(y_data):
            self.scan_canvas.create_text(w/2, h/2, text="[ NO SIGNAL DETECTED ]", fill="#39ff14", font=("Courier New", 14, "bold"))
            return

        pad_x, pad_y = 50, 30
        min_x, max_x = min(waves), max(waves)
        min_y, max_y = min(y_data), max(y_data)
        if max_x == min_x: max_x = min_x + 1
        if max_y == min_y: max_y = min_y + 1

        coords = []
        for x_val, y_val in zip(waves, y_data):
            cx = pad_x + (x_val - min_x) / (max_x - min_x) * (w - 2 * pad_x)
            cy = h - pad_y - (y_val - min_y) / (max_y - min_y) * (h - 2 * pad_y)
            coords.extend([cx, cy])

        self.scan_canvas.create_line(*coords, fill="#39ff14", width=2, joinstyle=tk.ROUND)
        self.scan_canvas.create_rectangle(pad_x, pad_y, w-pad_x, h-pad_y, outline="#207a20")

        font_style = ("Courier New", 10)
        self.scan_canvas.create_text(pad_x, h-10, text=f"{min_x:.0f}nm", fill="#207a20", font=font_style)
        self.scan_canvas.create_text(w-pad_x, h-10, text=f"{max_x:.0f}nm", fill="#207a20", font=font_style)
        self.scan_canvas.create_text(25, pad_y, text=f"{max_y:.2f}", fill="#207a20", font=font_style)
        self.scan_canvas.create_text(25, h-pad_y, text=f"{min_y:.2f}", fill="#207a20", font=font_style)
        title_map = {'intensity': "RAW INTENSITY", 'absorbance': "ABSORBANCE (A)", 'reflectance': "REFLECTANCE (%)"}
        self.scan_canvas.create_text(w-pad_x-10, pad_y+15, text=f"[{title_map[self.graph_mode]}]", fill="#39ff14", font=("Courier New", 10, "bold"), anchor="e")

    # ── Absorbance extraction ────────────────────────────────────────
    def _extract_absorbances(self, results):
        """Extract absorbance array from scan results dict.
        Centralizes the computation duplicated across scan/predict/train paths.
        """
        waves = results.get("wavelength", [])
        intensities = results.get("intensity", [])
        references = results.get("reference", [])
        if not waves or not intensities:
            return None

        r_raw_list = []
        for i in range(len(waves)):
            I = intensities[i] if i < len(intensities) else 0
            ref = max(references[i] if i < len(references) else 1, 1)
            r_raw_list.append(I / ref)

        valid_rs = [r for r in r_raw_list if r > 0]
        max_valid_abs = -math.log10(min(valid_rs)) if valid_rs else 4.0
        absorbances = [-math.log10(r) if r > 0 else max_valid_abs for r in r_raw_list]
        return absorbances

    # ── Hardware scan task ───────────────────────────────────────────
    def _hardware_scan_task(self):
        lamp_was_turned_on = False
        try:
            if not (self.hardware_connected and self.nirs is not None):
                return

            # --- [1. 扫描序列] ---
            self.root.after(0, lambda: self.status_display_var.set("WAKING UP HARDWARE..."))
            if hasattr(self.nirs, 'set_hibernate'): self.nirs.set_hibernate(False)
            time.sleep(0.5)

            self.root.after(0, lambda: self.status_display_var.set("IGNITING LAMP..."))
            self.nirs.set_lamp_on_off(True)
            lamp_was_turned_on = True
            time.sleep(1.5)

            self.root.after(0, lambda: self.status_display_var.set("CLEARING FAULTS..."))
            if hasattr(self.nirs, 'resetErrorStatus'): self.nirs.resetErrorStatus()
            elif hasattr(self.nirs, 'clear_error_status'): self.nirs.clear_error_status()
            time.sleep(0.2)

            self.root.after(0, lambda: self.status_display_var.set("RELOADING CONFIG..."))
            try:
                start_nm = int(self.wave_start_var.get())
                end_nm = int(self.wave_end_var.get())
            except (ValueError, TypeError):
                start_nm, end_nm = 900, 1700
            scan_type = 1 if self.scan_combo.get() == "Hadamard" else 0
            pga_val = self._parse_pga_gain()

            self.nirs.set_config(scanConfigIndex=8, scan_type=scan_type, num_patterns=228, num_repeats=6, wavelength_start_nm=start_nm, wavelength_end_nm=end_nm, width_px=7)
            self.nirs.set_pga_gain(pga_val)
            time.sleep(0.2)

            # --- [2. 首次扫描 - 带验证与重试] ---
            self.root.after(0, lambda: self.status_display_var.set("ACQUIRING DATA..."))
            success, results = self.nirs.scan_collect(num_repeats=1, max_retries=1)

            if not success:
                # Recovery: clear errors, re-apply config, retry once
                print("[HW_DIAG] First scan attempt failed — attempting recovery...")
                self.root.after(0, lambda: self.status_display_var.set("RECOVERING..."))
                if hasattr(self.nirs, 'clear_error_status'): self.nirs.clear_error_status()
                elif hasattr(self.nirs, 'resetErrorStatus'): self.nirs.resetErrorStatus()
                time.sleep(0.2)
                self.nirs.set_config(scanConfigIndex=8, scan_type=scan_type, num_patterns=228, num_repeats=6, wavelength_start_nm=start_nm, wavelength_end_nm=end_nm, width_px=7)
                self.nirs.set_pga_gain(pga_val)
                time.sleep(0.3)
                self.root.after(0, lambda: self.status_display_var.set("ACQUIRING DATA (RETRY)..."))
                success, results = self.nirs.scan_collect(num_repeats=1, max_retries=1)

            # --- [2.5 彻底失败提示] ---
            if not success:
                print("[HW_DIAG] CRITICAL: Both scan attempts failed — "
                      "device may need power cycle")
                self.root.after(0, lambda: messagebox.showerror(
                    "Scan Failed",
                    "Scan returned invalid data after multiple retries.\n\n"
                    "The device may need a power cycle.\n"
                    "Please disconnect and reconnect the NIRScan Nano."))

            self.root.after(0, lambda: self.status_display_var.set("ANALYZING DATA..."))

            intensities = results.get("intensity", [])
            if intensities:
                print(f"[HW_DIAG] Intensity: {len(intensities)} pts, range [{min(intensities)} ~ {max(intensities)}], ptp={max(intensities)-min(intensities)}")
            else:
                print("[HW_DIAG] WARNING: No intensity data in scan results!")

            scan_count = 1
            scan_interval = 0
            action = getattr(self, 'current_action', 'scan')
            try:
                scan_count = int(self.scan_count_var.get())
            except (ValueError, TypeError):
                scan_count = 1
            try:
                scan_interval = float(self.scan_interval_var.get())
            except (ValueError, TypeError):
                scan_interval = 0

            # --- Batch accumulation state ---
            self._batch_absorbances = []
            self._batch_intensities = []
            self._batch_mode = (action in ('train', 'predict')) and scan_count > 1

            # --- [3. 首次扫描结果处理] ---
            if self._batch_mode:
                absorbances = self._extract_absorbances(results)
                if absorbances:
                    self._batch_absorbances.append(absorbances)
                    self._batch_intensities.append(intensities)
                self.root.after(0, lambda: self.status_display_var.set(
                    f"ACQUIRING... (1/{scan_count})"))
            else:
                self._current_scan_suffix = f"_1" if scan_count > 1 else ""
                self.root.after(0, self._update_ui_with_results, results)

            # --- [4. 多次扫描循环 (带失败熔断)] ---
            for scan_idx in range(1, scan_count):
                if not success:
                    print(f"[HW_DIAG] Multi-scan aborted: previous scan failed, skipping scan {scan_idx+1}")
                    break

                if scan_interval > 0:
                    time.sleep(scan_interval)

                self.root.after(0, lambda: self.status_display_var.set("ACQUIRING DATA..."))
                success, results = self.nirs.scan_collect(num_repeats=1, max_retries=1)

                self.root.after(0, lambda: self.status_display_var.set("ANALYZING DATA..."))

                if not success:
                    print(f"[HW_DIAG] WARNING: Scan {scan_idx+1} returned no valid data")

                if self._batch_mode:
                    absorbances = self._extract_absorbances(results)
                    if absorbances:
                        self._batch_absorbances.append(absorbances)
                        self._batch_intensities.append(
                            results.get("intensity", []))
                    self.root.after(0, lambda i=scan_idx+1, t=scan_count:
                        self.status_display_var.set(
                            f"ACQUIRING... ({i}/{t})"))
                else:
                    self._current_scan_suffix = f"_{scan_idx + 1}"
                    self.root.after(0, self._update_ui_with_results, results)

            # --- [4.5 批量分发 (Batch Dispatch)] ---
            if self._batch_mode and self._batch_absorbances:
                self.root.after(0, self._dispatch_batch_results, action)

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Fatal Pipeline Failure", msg))
            self.is_scanning = False
            self.root.after(0, lambda: self._finalize_ui(getattr(self, 'current_action', 'scan')))
        finally:
            # --- [5. 保证清理：无论成功/失败/崩溃都关灯休眠] ---
            if lamp_was_turned_on:
                self._cleanup_gen = self._scan_gen  # capture gen for race guard
                threading.Thread(target=self._async_hardware_cleanup, daemon=True).start()

    # ── UI finalize ──────────────────────────────────────────────────
    def _finalize_ui(self, action=None):
        """全局智能收尾与状态机重置"""
        action = action or getattr(self, 'current_action', 'scan')

        # 1. 恢复系统状态指示灯
        self.is_scanning = False
        hw_status = "SLEEP" if self.hardware_connected else "DISCONN"
        self._update_ui_state("READY", hw_status, "#00ff00" if self.hardware_connected else "#ff4444")

        # 2. 重置大屏文字状态
        is_batch = getattr(self, '_batch_mode', False)
        if action == 'scan' or not is_batch:
            if hasattr(self, 'status_display_var'):
                self.status_display_var.set("[ READY ]")

        # 3. 按钮解锁
        if hasattr(self, 'scan_scan_btn'):
            self.scan_scan_btn.config(state=tk.NORMAL)
        if hasattr(self, 'start_predict_btn'):
            self.start_predict_btn.config(state=tk.NORMAL)

    # ── Batch dispatch (train / predict) ─────────────────────────────
    def _dispatch_batch_results(self, action):
        """Dispatch accumulated batch results once for train or predict."""
        if action == 'train':
            # Store batch data for the training worker
            with self.scan_data_lock:
                self.scan_data['batch_absorbances'] = self._batch_absorbances
                self.scan_data['batch_intensities'] = self._batch_intensities
                self.scan_data['batch_mode'] = True
            self.root.after(0, self._finalize_ui, action)
            threading.Thread(target=self._batch_train_worker, daemon=True).start()

        elif action == 'predict':
            self._batch_predict()
            self.root.after(0, self._finalize_ui, action)

    def _batch_predict(self):
        """Run prediction on all accumulated batch spectra, aggregate results."""
        from collections import Counter

        batch_absorbances = getattr(self, '_batch_absorbances', [])
        if not batch_absorbances:
            self.predict_result_var.set("ERROR:\nNO BATCH DATA")
            return

        try:
            task = self.tasks[self.current_task_idx]
            algorithm = task.get("algorithm", "")
            model_path = task.get("model_path")
            loaded_obj = joblib.load(model_path)

            if isinstance(loaded_obj, dict) and "classifier" in loaded_obj:
                model = loaded_obj["classifier"]
                hierarchical = loaded_obj.get("hierarchical")
            else:
                model = loaded_obj
                hierarchical = None

            task_type = task.get("task_type", "Classification")
            all_predictions = []
            all_confidences = []

            for abs_data in batch_absorbances:
                X_filtered, _ = self._preprocess_spectrum(
                    np.array(abs_data), algorithm)
                if X_filtered is None:
                    continue

                if task_type == "Classification":
                    if hierarchical and hierarchical.get("clf2") is not None:
                        self._hier_level_map = hierarchical
                        pred = self._hierarchical_predict(X_filtered)
                    else:
                        pred = model.predict(X_filtered)
                    all_predictions.append(str(pred[0]))
                    conf, source = self._compute_confidence(model, X_filtered)
                    all_confidences.append(conf)
                    if not hasattr(self, '_batch_conf_source'):
                        self._batch_conf_source = source
                else:
                    pred = model.predict(X_filtered)
                    all_predictions.append(float(np.squeeze(pred)))

            if not all_predictions:
                self.predict_result_var.set("ERROR:\nALL PREDICTIONS FAILED")
                return

            # Capture confidence source from first scan (same model → same source)
            confidence_source = getattr(self, '_batch_conf_source', "vote")

            # Aggregate results
            if task_type == "Classification":
                vote_counts = Counter(all_predictions)
                final_result = vote_counts.most_common(1)[0][0]
                votes_for = vote_counts.most_common(1)[0][1]
                avg_confidence = np.mean(all_confidences)

                # Threshold check
                accepted = avg_confidence >= self.CONFIDENCE_THRESHOLD
                if not accepted:
                    final_result = "UNKNOWN"
                    display_text = (f"[ BATCH ANALYSIS COMPLETE ]\n\n"
                                    f"RESULT: UNKNOWN\n"
                                    f"CONF:  {avg_confidence:.1f}% "
                                    f"(<{self.CONFIDENCE_THRESHOLD:.0f}%)\n"
                                    f"BATCH: {len(all_predictions)} SCANS")
                    detail = (f"REJECTED: avg confidence {avg_confidence:.1f}% "
                              f"below threshold {self.CONFIDENCE_THRESHOLD:.0f}%")
                else:
                    display_text = (f"[ BATCH ANALYSIS COMPLETE ]\n\n"
                                    f"RESULT: {final_result}\n"
                                    f"CONF:  {avg_confidence:.1f}%\n"
                                    f"BATCH: {len(all_predictions)} SCANS")
                    detail = (f"MAJORITY: {final_result} "
                              f"({votes_for}/{len(all_predictions)})")
                self.predict_detail_var.set(detail)

                # Log batch prediction
                self._log_prediction_metrics(
                    task, final_result, avg_confidence, confidence_source,
                    accepted=accepted, mode="batch",
                    batch_size=len(all_predictions),
                    hierarchical=(hierarchical and hierarchical.get("clf2") is not None),
                    vote_counts=dict(vote_counts))
            else:
                final_val = np.mean(all_predictions)
                std_val = np.std(all_predictions)

                try:
                    meta = json.loads(task.get("description", "{}"))
                    target_name = meta.get("reg_target", "VALUE").upper()
                except Exception:
                    target_name = "VALUE"

                display_text = f"[ BATCH ANALYSIS COMPLETE ]\n\n"
                display_text += f"{target_name}:\n"
                display_text += f"MEAN: {final_val:.3f}\n"
                display_text += f"STD:  {std_val:.4f}\n"
                display_text += f"BATCH: {len(all_predictions)} SCANS"

                self.predict_detail_var.set(
                    f"Mean of {len(all_predictions)} scans")

            self.predict_result_var.set(display_text)

            # ── Load and display model performance metrics ────────────
            if task_type == "Classification":
                task = self.tasks[self.current_task_idx]
                perf = self._load_latest_perf(task)
                perf_text = self._format_perf_text(perf)
                self.predict_perf_var.set(perf_text)

        except Exception as e:
            print(f"[BATCH PREDICT ERROR] {e}")
            self.predict_result_var.set("BATCH INFERENCE\nFAILURE")
            self.predict_detail_var.set(f"Error: {str(e)[:50]}")

    # ── Hardware cleanup ─────────────────────────────────────────────
    def _async_hardware_cleanup(self):
        """后台静默处理关机，尝试解除设备异常状态后关灯休眠。
        带竞态保护：如果新扫描已开始，跳过关灯与休眠操作。"""
        try:
            print("[SYS] Performing async hardware shutdown...")
            if self.nirs:
                # Step 1: Attempt to clear any stuck error state before shutting down
                try:
                    if hasattr(self.nirs, 'clear_error_status'):
                        self.nirs.clear_error_status()
                    elif hasattr(self.nirs, 'resetErrorStatus'):
                        self.nirs.resetErrorStatus()
                except Exception:
                    pass  # best-effort, don't block lamp-off

                # Step 2: Race guard — use generation counter to detect
                # if a NEWER scan has started since this cleanup was launched.
                # (is_scanning is still True for the scan that launched this
                #  cleanup, so we can't use it as a guard.)
                my_gen = getattr(self, '_cleanup_gen', -1)
                cur_gen = getattr(self, '_scan_gen', 0)
                if cur_gen != my_gen:
                    print(f"[SYS] Newer scan (gen {cur_gen}) detected "
                          f"during cleanup (gen {my_gen}), "
                          f"skipping lamp-off + hibernate")
                    return

                # Step 3: Turn off lamp (always, unless guarded above)
                self.nirs.set_lamp_on_off(False)

                # Step 4: Hibernate the device
                if hasattr(self.nirs, 'set_hibernate'):
                    self.nirs.set_hibernate(True)

                print("[SYS] Async hardware shutdown complete")
        except Exception as e:
            print(f"[HW_WARN] Hardware cleanup failed: {e}")

    # ── CSV save ─────────────────────────────────────────────────────
    def _save_scan_data_to_csv(self, results):
        try:
            waves = results.get("wavelength", [])
            intensities = results.get("intensity", [])
            references = results.get("reference", [])
            if not waves or not intensities:
                return

            r_raw_list = []
            r_pct_list = []
            for i in range(len(waves)):
                I = intensities[i] if i < len(intensities) else 0
                ref = max(references[i] if i < len(references) else 1, 1)
                r_raw_list.append(I / ref)
                r_pct_list.append((I / ref) * 100)

            valid_rs = [r for r in r_raw_list if r > 0]
            max_valid_abs = -math.log10(min(valid_rs)) if valid_rs else 4.0
            a_list = [-math.log10(r) if r > 0 else max_valid_abs for r in r_raw_list]

            # --- 构建保存路径 (带权限容错回退) ---
            save_dir = self.save_path_var.get().strip()
            if not save_dir:
                save_dir = "data/spectra/"

            filename = self.save_filename_var.get().strip()
            suffix = getattr(self, '_current_scan_suffix', '')
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"spectrum_{timestamp}"

            filepath = os.path.join(save_dir, f"{filename}{suffix}.csv")

            def _write_csv(target_path):
                """原子化写入：先写临时文件，再 rename，防止写一半断电损坏"""
                tmp_path = target_path + ".tmp"
                with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["wavelength", "intensity", "reflectance", "absorbance"])
                    for i in range(len(waves)):
                        writer.writerow([waves[i], intensities[i], r_pct_list[i], a_list[i]])
                os.replace(tmp_path, target_path)

            # 尝试 1：写入用户指定的路径
            try:
                os.makedirs(save_dir, exist_ok=True)
                _write_csv(filepath)
                print(f"[SYS] Spectrum data saved to: {filepath}")
                return
            except PermissionError:
                pass  # 权限不足，进入回退逻辑

            # 尝试 2：回退到 HOME 目录下的安全路径
            fallback_dir = os.path.join(os.path.expanduser("~"), "spectra_data")
            try:
                os.makedirs(fallback_dir, exist_ok=True)
                fallback_path = os.path.join(fallback_dir, f"{filename}{suffix}.csv")
                _write_csv(fallback_path)
                print(f"[SYS] Spectrum data saved to fallback: {fallback_path}")
                self.root.after(0, lambda: messagebox.showwarning(
                    "Save Path Fallback",
                    f"Permission denied for:\n{filepath}\n\n"
                    f"Data saved to fallback location:\n{fallback_path}"
                ))
            except Exception as e2:
                print(f"[SYS] Failed to save spectrum CSV (fallback also failed): {e2}")
                self.root.after(0, lambda: messagebox.showerror(
                    "Save Failed",
                    f"Could not save spectrum data.\n\n"
                    f"Primary path: {filepath}\n"
                    f"Error: Permission denied\n\n"
                    f"Fallback path also failed: {e2}"
                ))
        except Exception as e:
            print(f"[SYS] Failed to save spectrum CSV: {e}")

    # ── Error handling ───────────────────────────────────────────────
    def _handle_error(self, err_msg):
        self.is_scanning = False
        # 删掉 self.progress_bar.pack_forget()
        self.status_display_var.set(f"[ FATAL ERROR ]\n\n{err_msg}")
        self._update_ui_state("HW ERROR", "DISCONN", "#ff4444")

    # ── Data processing ──────────────────────────────────────────────
    def _process_scan_data(self, results):
        waves = results.get("wavelength", [])
        intensities = results.get("intensity", [])
        references = results.get("reference", [])
        if not waves or not intensities: return

        r_raw_list, r_pct_list = [], []
        for i in range(len(waves)):
            I = intensities[i] if i < len(intensities) else 0
            ref = max(references[i] if i < len(references) else 1, 1)
            r_raw_list.append(I / ref)
            r_pct_list.append((I / ref) * 100)

        valid_rs = [r for r in r_raw_list if r > 0]
        max_valid_abs = -math.log10(min(valid_rs)) if valid_rs else 4.0

        a_list = [-math.log10(r) if r > 0 else max_valid_abs for r in r_raw_list]
        with self.scan_data_lock:
            self.scan_data = {'wave': waves, 'intensity': intensities, 'reflectance': r_pct_list, 'absorbance': a_list}
        self._draw_graph()

    # ── UI result dispatch ───────────────────────────────────────────
    def _update_ui_with_results(self, results):
        action = getattr(self, 'current_action', 'scan')

        temp_val = results.get("temperature_system")
        if temp_val is not None and hasattr(self, 'temp_label'):
            self.root.after(0, lambda t=temp_val: self.temp_label.config(text=f"TEMP:{t:.1f}°C"))

        if action == 'scan':
            self._process_scan_data(results)
            if hasattr(self, 'status_display_var'):
                self.status_display_var.set("[ SCAN COMPLETE ]\nData Acquired")
            self._save_scan_data_to_csv(results)
            self.root.after(0, self._finalize_ui, action)

        elif action == 'predict':
            try:
                waves = results.get("wavelength", [])
                intensities = results.get("intensity", [])
                references = results.get("reference", [])
                if not waves or not intensities:
                    self.predict_result_var.set("ERROR:\nNO SCAN DATA")
                    return

                r_raw_list = []
                for i in range(len(waves)):
                    I = intensities[i] if i < len(intensities) else 0
                    ref = max(references[i] if i < len(references) else 1, 1)
                    r_raw_list.append(I / ref)

                valid_rs = [r for r in r_raw_list if r > 0]
                max_valid_abs = -math.log10(min(valid_rs)) if valid_rs else 4.0
                absorbances = [-math.log10(r) if r > 0 else max_valid_abs for r in r_raw_list]

                task = self.tasks[self.current_task_idx]
                algorithm = task.get("algorithm", "")
                model_path = task.get("model_path")
                loaded_obj = joblib.load(model_path)

                try: meta = json.loads(task.get("description", "{}"))
                except: meta = {}

                if isinstance(loaded_obj, dict) and "classifier" in loaded_obj:
                    model = loaded_obj["classifier"]
                    hierarchical = loaded_obj.get("hierarchical")
                else:
                    model = loaded_obj
                    hierarchical = None

                X_filtered, _ = self._preprocess_spectrum(absorbances, algorithm)
                if X_filtered is None:
                    self.predict_result_var.set("ERROR:\nPREPROCESS FAILED")
                    return

                task_type = task.get("task_type", "Classification")

                if task_type == "Classification":
                    # Try hierarchical prediction first when available
                    is_hier = (hierarchical and hierarchical.get("clf2") is not None)
                    if is_hier:
                        self._hier_level_map = hierarchical
                        pred = self._hierarchical_predict(X_filtered)
                        final_result = str(pred[0])
                    else:
                        pred = model.predict(X_filtered)
                        final_result = str(pred[0])

                    confidence, source = self._compute_confidence(model, X_filtered)

                    # Threshold check: reject low-confidence predictions
                    accepted = confidence >= self.CONFIDENCE_THRESHOLD
                    if not accepted:
                        raw_result = final_result
                        final_result = "UNKNOWN"
                        display_text = (f"[ ANALYSIS COMPLETE ]\n\n"
                                        f"RESULT: UNKNOWN\n"
                                        f"CONF: {confidence:.1f}% "
                                        f"(<{self.CONFIDENCE_THRESHOLD:.0f}%)")
                        if hasattr(self, 'predict_detail_var'):
                            self.predict_detail_var.set(
                                f"REJECTED: confidence {confidence:.1f}% below "
                                f"threshold {self.CONFIDENCE_THRESHOLD:.0f}% "
                                f"({source})"
                                + (" [hierarchical]" if is_hier else ""))
                    else:
                        raw_result = final_result
                        display_text = (f"[ ANALYSIS COMPLETE ]\n\n"
                                        f"RESULT: {final_result}\n"
                                        f"CONF: {confidence:.1f}%")
                        if hasattr(self, 'predict_detail_var'):
                            self.predict_detail_var.set(
                                f"CONFIDENCE: {confidence:.1f}% ({source})"
                                + (" [hierarchical]" if is_hier else ""))

                    # Log prediction metrics to inference file
                    self._log_prediction_metrics(
                        task, final_result, confidence, source,
                        accepted=accepted, mode="single",
                        hierarchical=is_hier)
                else:
                    final_val = float(np.squeeze(pred))
                    target_name = meta.get("reg_target", "VALUE").upper()
                    display_text = f"[ ANALYSIS COMPLETE ]\n\n{target_name}:\n{final_val:.3f}"

                    reg_range = meta.get("reg_range", "")
                    if reg_range:
                        self.predict_detail_var.set(f"Valid Range: [{reg_range}]")
                    else:
                        self.predict_detail_var.set("Target successfully quantified.")

                self.predict_result_var.set(display_text)

            except Exception as e:
                print(f"[PREDICT FATAL ERROR] {e}")
                self.predict_result_var.set("INFERENCE\nFAILURE")
                self.predict_detail_var.set("System Error Occurred")
            finally:
                self.root.after(0, self._finalize_ui, action)

        elif action == 'train':
            waves = results.get("wavelength", [])
            intensities = results.get("intensity", [])
            references = results.get("reference", [])
            if not waves or not intensities: return

            r_raw_list = []
            for i in range(len(waves)):
                I = intensities[i] if i < len(intensities) else 0
                ref = max(references[i] if i < len(references) else 1, 1)
                r_raw_list.append(I / ref)

            valid_rs = [r for r in r_raw_list if r > 0]
            max_valid_abs = -math.log10(min(valid_rs)) if valid_rs else 4.0
            absorbances = [-math.log10(r) if r > 0 else max_valid_abs for r in r_raw_list]

            train_data = {'wave': waves, 'intensity': intensities, 'absorbance': absorbances}
            with self.scan_data_lock:
                self.scan_data = train_data

            self.root.after(0, self._finalize_ui, action)
            threading.Thread(target=self._train_model_worker, daemon=True).start()
