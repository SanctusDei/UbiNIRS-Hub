import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as messagebox
import threading
import time
import math  
import numpy as np
import joblib
import scipy.signal
import socket
import subprocess
import sqlite3
import os
import shutil
from INA219 import INA219
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_decomposition import PLSRegression
from NIRS import NIRS
from functools import partial
import json
import csv
import pandas as pd
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
import collections

from gui_scan import ScanWorkflowMixin
from gui_ml import MLEngineMixin
from gui_tasks import TaskManagerMixin

class SpectrometerApp(ScanWorkflowMixin, MLEngineMixin, TaskManagerMixin):
    def __init__(self, root):
      
        self.root = root
        self.root.title("UbiNIRS-Hub v2.0")
        
        self.train_tag_var = tk.StringVar(value="")
        self.train_target_var = tk.StringVar(value="")
        self.train_task_name_var = tk.StringVar(value="TASK: N/A")
        self.train_task_type_var = tk.StringVar(value="TYPE: N/A")
        # 5-inch DSI display fullscreen config
        self.root.geometry("800x480+0+0") 
        self.root.attributes('-fullscreen', True) 
        self.root.bind("<Escape>", lambda e: self.root.attributes('-fullscreen', False))
        self._init_db()
        
        self.root.config(bg="#404040", cursor="none")
        
        style = ttk.Style()
        style.theme_use('classic')

        self.SCREEN_BG = "#9e9e9e"
        self.SCREEN_FG = "#1a1a1a"
        self.PANEL_BG = "#757575"
        self.BTN_BG = "#b5b5b5"
        self.FONT_SYS = ("Courier New", 12, "bold")
        self.FONT_TITLE = ("Courier New", 16, "bold")
        
        # Hardware state machine
        self.nirs = None
        self.hardware_connected = False
        self.is_scanning = False
        self._scan_gen = 0   # generation counter for cleanup race guard
        self._nirs_lock = threading.Lock()  # protects self.nirs across init/heartbeat threads
        self.status_display_var = tk.StringVar(value="[ READY ]\n\nSELECT A TASK TO BEGIN")
        self.predict_result_var = tk.StringVar(value="READY.\nAWAITING SCAN.")
        self.scan_data_lock = threading.Lock()
        self.available_predict_tasks = []
        self.is_configuring = False
        self.ml_models = {}

        # Setup warning string variables for dynamic UI lockouts
        self.execute_warn_var = tk.StringVar()
        self.config_warn_var = tk.StringVar()
        self.scan_warn_var = tk.StringVar()
        
        try:
            self.power_sensor = INA219(addr=0x42)
            print("[SYS] INA219 电池传感器已连接")
        except Exception as e:
            self.power_sensor = None
            print(f"[WARN] 找不到 INA219 传感器: {e}")
            
        self.battery_warned = False # 用于记录是否已经警告过，防止重复弹窗

        # Coulomb counter state (integrates INA219 current over time)
        self._battery_consumed_mah = 0.0
        self._battery_last_ts = time.time()
        self._battery_capacity_mah = 1800.0  # 2S LiPo rated capacity
        
        self._load_models()
        self.setup_ui()

        # Ensure clean shutdown when the window is closed (X button, WM close, etc.)
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        self.is_booting = True
        
        # Async hardware init & heartbeat
        threading.Thread(target=self._init_hardware_safely, daemon=True).start()
        threading.Thread(target=self._hardware_heartbeat_loop, daemon=True).start()
        
    # 将这个方法放到你的类中
       
        
        
    def _get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "NO NETWORK"
    
            
    def _refresh_ip_cache(self):
        """后台异步刷新 IP 缓存，绝不阻塞主线程"""
        def _fetch():
            self._cached_ip = self._get_local_ip()
        threading.Thread(target=_fetch, daemon=True).start()
        self.root.after(30000, self._refresh_ip_cache)  # 每30秒刷新一次

    def _on_window_close(self):
        """Clean shutdown when the window is closed via WM or X button.
        Explicitly closes the NIRS C++ object and unregisters its atexit
        handler to prevent the "Cleanning up" message on clean exit."""
        print("[SYS] Window close requested — shutting down...")
        with self._nirs_lock:
            if self.nirs is not None:
                if hasattr(self.nirs, 'close'):
                    self.nirs.close()
                self.nirs = None
        self.root.destroy()

    def _update_ui_state(self, sys_status="READY", hw_status="UNKNOWN", color="#8a8a8a"):
        # 🐛 Bug Fix: 使用缓存 IP，避免每次同步网络请求阻塞主线程导致 XIO error 11
        local_ip = getattr(self, '_cached_ip', 'LOADING...')
        self.status_label.config(text=f"SYS: {sys_status}\nHW : [{hw_status}]\nIP : {local_ip}", fg="#111111", bg=color)
        
        # 动态 UI 熔断机制：断联时全局变灰并弹出红字警告

        if not self.hardware_connected and sys_status != "BOOTING...":
            warn_msg = "⚠ ERROR: DEVICE DISCONNECTED"
            if hasattr(self, 'execute_warn_var'): self.execute_warn_var.set(warn_msg)
            if hasattr(self, 'config_warn_var'): self.config_warn_var.set(warn_msg)
            if hasattr(self, 'scan_warn_var'): self.scan_warn_var.set(warn_msg)
            
            # Disable all actionable buttons
            if hasattr(self, 'big_scan_btn'): self.big_scan_btn.config(state=tk.DISABLED, bg="#4d4d4d", fg="#888888")
            if hasattr(self, 'scan_scan_btn'): self.scan_scan_btn.config(state=tk.DISABLED, bg="#4d4d4d", fg="#888888")
            if hasattr(self, 'apply_btn'): self.apply_btn.config(state=tk.DISABLED, bg="#4d4d4d", fg="#888888")
            if hasattr(self, 'lamp_btn'): self.lamp_btn.config(state=tk.DISABLED)
            if hasattr(self, 'reset_btn'): self.reset_btn.config(state=tk.DISABLED, bg="#4d4d4d", fg="#888888")
        else:
            # Clear warnings
            if hasattr(self, 'execute_warn_var'): self.execute_warn_var.set("")
            if hasattr(self, 'config_warn_var'): self.config_warn_var.set("")
            if hasattr(self, 'scan_warn_var'): self.scan_warn_var.set("")
            
            # Enable buttons gracefully
            if hasattr(self, 'big_scan_btn') and not self.is_scanning: 
                self.big_scan_btn.config(state=tk.NORMAL, bg="#a3a3a3", fg="#111111")
            if hasattr(self, 'scan_scan_btn') and not self.is_scanning: 
                self.scan_scan_btn.config(state=tk.NORMAL, bg="#a3a3a3", fg="#1a1a1a")
            if hasattr(self, 'apply_btn') and not self.is_configuring: 
                self.apply_btn.config(state=tk.NORMAL, bg="#a3a3a3", fg=self.SCREEN_FG)
            if hasattr(self, 'lamp_btn'): 
                self.lamp_btn.config(state=tk.NORMAL)
            if hasattr(self, 'reset_btn'): 
                self.reset_btn.config(state=tk.NORMAL, bg="#8b0000", fg="#ffffff")

    def _init_hardware_safely(self):
        try:
            self.root.after(0, lambda: self._update_ui_state("BOOTING...", "WAKING", "#ffff00"))

            # Redirect C++ stdout (fd 1) to /dev/null during constructor.
            # sys.stdout redirect only catches Python print(); C++ std::cout
            # writes to fd 1 directly, requiring OS-level redirection.
            import os as _os
            fd_devnull = _os.open(_os.devnull, _os.O_WRONLY)
            fd_stdout_saved = _os.dup(1)  # save fd 1
            _os.dup2(fd_devnull, 1)       # redirect fd 1 → /dev/null
            _os.close(fd_devnull)
            try:
                self.nirs = NIRS()
            finally:
                _os.dup2(fd_stdout_saved, 1)  # restore fd 1
                _os.close(fd_stdout_saved)

            # Check if USB actually opened — if not, retry once after a short delay
            usb_open_failed = (self.nirs is None)
            if not usb_open_failed:
                # --- DLPC Boot Wait Loop ---
                # After deep hibernation, the DLPC150 DLP controller needs
                # 3-10 seconds to boot its flash image, while the Tiva MCU
                # wakes up much faster.  The C++ constructor's internal
                # retry loop may not wait long enough for the DLPC to fully
                # initialise.  We poll display_version() here to confirm
                # the device (Tiva + DLPC) is truly ready before proceeding
                # to scan-configuration commands that depend on the DLPC.
                MAX_DLPC_WAIT_SEC = 15.0
                DLPC_POLL_INTERVAL = 2.0
                dlpc_ready = False
                dlpc_wait_start = time.time()
                while time.time() - dlpc_wait_start < MAX_DLPC_WAIT_SEC:
                    try:
                        if hasattr(self.nirs, 'display_version'):
                            ver = self.nirs.display_version()
                            if ver is not None and ver >= 0:
                                dlpc_ready = True
                                break
                    except Exception:
                        pass
                    print(f"[SYS] DLPC not ready yet "
                          f"(elapsed {time.time() - dlpc_wait_start:.0f}s), "
                          f"waiting {DLPC_POLL_INTERVAL}s...")
                    time.sleep(DLPC_POLL_INTERVAL)

                if not dlpc_ready:
                    print(f"[SYS] DLPC did not become ready within "
                          f"{MAX_DLPC_WAIT_SEC:.0f}s — retrying full init...")
                    usb_open_failed = True
                else:
                    print(f"[SYS] DLPC ready after "
                          f"{time.time() - dlpc_wait_start:.0f}s")

            if usb_open_failed:
                print("[SYS] USB open failed on first attempt, retrying after 2s...")
                time.sleep(2.0)
                fd_devnull2 = _os.open(_os.devnull, _os.O_WRONLY)
                fd_saved2 = _os.dup(1)
                _os.dup2(fd_devnull2, 1)
                _os.close(fd_devnull2)
                try:
                    self.nirs = NIRS()
                finally:
                    _os.dup2(fd_saved2, 1)
                    _os.close(fd_saved2)

            # Guard: if USB still isn't open after retries, bail out
            if self.nirs is None:
                raise RuntimeError("NIRS constructor returned None")

            # Wake the device (may be redundant if constructor already did it,
            # but harmless — ensures device stays awake)
            if hasattr(self.nirs, 'set_hibernate'):
                self.nirs.set_hibernate(False)
            time.sleep(0.5)  # brief settle after wake command

            # Clear stale errors
            if hasattr(self.nirs, 'clear_error_status'):
                self.nirs.clear_error_status()

            # Re-apply config now that device is awake and DLPC is ready
            if hasattr(self.nirs, 'set_config'):
                self.nirs.set_config(scanConfigIndex=8, scan_type=1, num_patterns=228,
                                     num_repeats=6, wavelength_start_nm=900,
                                     wavelength_end_nm=1700, width_px=7)
                self.nirs.set_pga_gain(0)  # 0 = Auto
                self.nirs.set_lamp_on_off(False)

            self.hardware_connected = True

            # --- Debug: print device status & parameters after init ---
            print("[DEBUG] === Hardware Init Complete ===")
            print(f"[DEBUG] USB Connection: OK")
            print(f"[DEBUG] Config Applied: scanConfigIndex=8, scan_type=Hadamard(1), num_patterns=228, num_repeats=6")
            print(f"[DEBUG] Wavelength: 900-1700 nm, Width: 7 px")
            print(f"[DEBUG] PGA Gain: Auto(0), Lamp: OFF")
            print(f"[DEBUG] Status: hardware_connected=True, nirs={self.nirs}")

            self.root.after(0, lambda: self._update_ui_state("READY", "READY", "#00ff00"))
        except Exception:
            with self._nirs_lock:
                if self.nirs is not None:
                    if hasattr(self.nirs, 'close'): self.nirs.close()
                self.nirs = None
            self.hardware_connected = False
            print("[DEBUG] === Hardware Init FAILED ===")
            self.root.after(0, lambda: self._update_ui_state("ERROR", "DISCONN", "#ff4444"))
        finally:
            self.is_booting = False

    def _hardware_heartbeat_loop(self):
        """Kernel-level heartbeat via lsusb for flawless disconnect capture (1.5 Hz = every 667 ms)"""
        while True:
            time.sleep(0.667)  # 1.5 Hz polling rate

            if getattr(self, 'is_booting', False) or getattr(self, 'is_scanning', False) or getattr(self, 'is_configuring', False):
                continue

            try:
                result = subprocess.run(['lsusb'], stdout=subprocess.PIPE, text=True)
                usb_list = result.stdout.lower()
                is_physically_plugged = "0451:" in usb_list or "texas" in usb_list or "stellaris" in usb_list

                if is_physically_plugged:
                    if not self.hardware_connected and self.nirs is None:
                        # Clean up any stale reference before creating new connection
                        with self._nirs_lock:
                            if getattr(self, 'nirs', None) is not None:
                                del self.nirs
                        # Suppress C++ stdout noise during reconnection
                        import os as _os
                        fd_devnull = _os.open(_os.devnull, _os.O_WRONLY)
                        fd_saved = _os.dup(1)
                        _os.dup2(fd_devnull, 1)
                        _os.close(fd_devnull)
                        try:
                            nirs_new = NIRS()
                        finally:
                            _os.dup2(fd_saved, 1)
                            _os.close(fd_saved)
                        with self._nirs_lock:
                            self.nirs = nirs_new
                        if self.nirs is not None:
                            # Wake device
                            if hasattr(self.nirs, 'set_hibernate'):
                                self.nirs.set_hibernate(False)
                            time.sleep(1.5)  # allow device to fully wake

                            if hasattr(self.nirs, 'clear_error_status'):
                                self.nirs.clear_error_status()

                            # Re-apply all user configuration from UI state
                            try:
                                start_nm = int(self.wave_start_var.get())
                                end_nm = int(self.wave_end_var.get())
                                scan_type = 1 if self.scan_combo.get() == "Hadamard" else 0
                                pga_val = self._parse_pga_gain()
                                is_lamp_on = self.lamp_var.get()

                                self.nirs.set_config(
                                    scanConfigIndex=8, scan_type=scan_type,
                                    num_patterns=228, num_repeats=6,
                                    wavelength_start_nm=start_nm,
                                    wavelength_end_nm=end_nm, width_px=7)
                                self.nirs.set_pga_gain(pga_val)
                                self.nirs.set_lamp_on_off(is_lamp_on)
                                print(f"[HB] Config re-applied after reconnect: "
                                      f"range={start_nm}-{end_nm}nm, "
                                      f"mode={'Hadamard' if scan_type==1 else 'Column'}, "
                                      f"pga={pga_val}, lamp={'ON' if is_lamp_on else 'OFF'}")
                            except Exception as cfg_err:
                                print(f"[HB] WARNING: Could not re-apply config: {cfg_err}")

                            # Health check: verify device is truly responsive
                            try:
                                if hasattr(self.nirs, 'display_version'):
                                    ver = self.nirs.display_version()
                                    if ver is not None and ver < 0:
                                        raise RuntimeError("display_version returned failure")
                            except Exception as probe_err:
                                print(f"[HB] Health check FAILED after reconnect: {probe_err}")
                                with self._nirs_lock:
                                    if self.nirs is not None:
                                        if hasattr(self.nirs, 'close'):
                                            self.nirs.close()
                                    self.nirs = None
                                continue  # skip to next heartbeat poll

                            self.hardware_connected = True
                            self.root.after(0, lambda: self._update_ui_state("READY", "READY", "#00ff00"))
                else:
                    if self.hardware_connected:
                        self.hardware_connected = False
                        with self._nirs_lock:
                            if self.nirs is not None:
                                if hasattr(self.nirs, 'close'): self.nirs.close()
                            self.nirs = None
                        self.root.after(0, lambda: self._update_ui_state("ERROR", "DISCONN", "#ff4444"))
            except Exception:
                pass

    def _force_wake_up_hardware(self):
        if not self.hardware_connected or self.nirs is None: return
        try:
            self.root.after(0, lambda: self._update_ui_state("WAKING...", "WAKING", "#ffff00"))
            if hasattr(self.nirs, 'set_hibernate'):
                self.nirs.set_hibernate(False)
                time.sleep(1.5)
            if hasattr(self.nirs, 'resetErrorStatus'):
                self.nirs.resetErrorStatus()
            elif hasattr(self.nirs, 'clear_error_status'):
                self.nirs.clear_error_status()
            time.sleep(0.2)

            # Health check: verify device is responsive after wake
            try:
                if hasattr(self.nirs, 'display_version'):
                    ver = self.nirs.display_version()
                    if ver is not None and ver < 0:
                        raise RuntimeError("display_version returned failure")
            except Exception as probe_err:
                print(f"[WAKE] Health check FAILED after wake: {probe_err}")
                self.hardware_connected = False
                with self._nirs_lock:
                    if self.nirs is not None:
                        if hasattr(self.nirs, 'close'):
                            self.nirs.close()
                    self.nirs = None
                self.root.after(0, lambda: self._update_ui_state("ERROR", "DISCONN", "#ff4444"))
                return

            self.root.after(0, lambda: self._update_ui_state("READY", "READY", "#00ff00"))
        except Exception:
            self.hardware_connected = False
            with self._nirs_lock:
                if self.nirs is not None:
                    if hasattr(self.nirs, 'close'): self.nirs.close()
                self.nirs = None
            self.root.after(0, lambda: self._update_ui_state("ERROR", "DISCONN", "#ff4444"))

    def _parse_pga_gain(self):
        pga_str = self.pga_combo.get()
        if pga_str == "Auto":
            return 0
        try:
            return int(pga_str)
        except (ValueError, TypeError):
            return 0

    def _load_models(self):
        try:
            pipeline_data = joblib.load("models/spectral_svm_pipeline.pkl")
            self.ml_models["Textile Classification"] = {
                "clf": pipeline_data['classifier'],
                "mask": pipeline_data['feature_mask']
            }
            print("[SYS] ML Model loaded successfully!")
        except FileNotFoundError:
            print("[SYS] WARN: Model file not found.")
        except Exception as e:
            print(f"[SYS] WARN: Load failed -> {str(e)}")
                
    def setup_ui(self):
    
        self.ctrl_frame = tk.Frame(self.root, bg=self.PANEL_BG, bd=5, relief=tk.RAISED)
        self.ctrl_frame.place(relx=0, rely=0, relwidth=0.30, relheight=1.0) 
        
        self.monitor_bezel = tk.Frame(self.root, bg="#505050", bd=8, relief=tk.RIDGE)
        self.monitor_bezel.place(relx=0.30, rely=0, relwidth=0.70, relheight=1.0)

        self.container = tk.Frame(self.monitor_bezel, bg=self.SCREEN_BG, bd=4, relief=tk.SUNKEN)
        self.container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.container.rowconfigure(0, weight=1)
        self.container.columnconfigure(0, weight=1)
          
        self.execute_page = tk.Frame(self.container, bg=self.SCREEN_BG)
        self.execute_page.grid(row=0, column=0, sticky="nsew")
        self.config_page = tk.Frame(self.container, bg=self.SCREEN_BG)
        self.config_page.grid(row=0, column=0, sticky="nsew")
        self.scan_page = tk.Frame(self.container, bg=self.SCREEN_BG)
        self.scan_page.grid(row=0, column=0, sticky="nsew")
        self.task_page = tk.Frame(self.container, bg=self.SCREEN_BG)
        self.task_page.grid(row=0, column=0, sticky="nsew")
        self.new_task_page = tk.Frame(self.container, bg=self.SCREEN_BG)
        self.new_task_page.grid(row=0, column=0, sticky="nsew")
        self.train_page = tk.Frame(self.container, bg=self.SCREEN_BG)
        self.train_page.grid(row=0, column = 0, sticky="nsew")
        self.predict_page = tk.Frame(self.container, bg=self.SCREEN_BG)
        self.predict_page.grid(row=0, column=0, sticky="nsew")
        
        self.build_left_panel()
        self.build_execute_panel()
        self.build_config_panel()
        self.build_scan_panel()
        self.build_task_panel()
        self.build_new_task_panel()
        self.build_train_panel()
        self.build_predict_panel()
        self.show_execute_page()
        
        
        
        # 开启异步 IP 缓存刷新（防止 _get_local_ip 阻塞主线程）
        self._refresh_ip_cache()

        #开启电量更新
        self._update_battery_status()
        
    def open_keyboard(self, target_var, title_text):
        """Standard Full QWERTY Keyboard - In-App Overlay (100% 保证不被遮挡)"""
        
        # 1. 如果当前已经有个键盘了，先销毁它
        if hasattr(self, '_current_kbd') and self._current_kbd is not None:
            try:
                self._current_kbd.destroy()
            except:
                pass

        # 放弃 Toplevel，直接在 root 主窗口内部创建一个 Frame！
        # 因为它属于主窗口的一部分，所以绝对不可能被主窗口自己遮挡。
        pad = tk.Frame(self.root, bg="#2b2b2b", bd=6, relief=tk.RAISED)
        self._current_kbd = pad
        
        # 绝对定位：宽度 800，高度 300，锚点在左下角 (sw)，放在坐标 (0, 480) 的位置
        pad.place(x=0, y=480, width=800, height=220, anchor="sw")
        
        # 强制把这个 Frame 提起到本窗口的所有控件之上
        pad.tkraise()

        # Header Frame
        top_frame = tk.Frame(pad, bg="#2b2b2b")
        top_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        tk.Label(top_frame, text=f"SET {title_text}:", bg="#2b2b2b", fg="#39ff14", font=("Courier New", 14, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        disp = tk.Label(top_frame, textvariable=target_var, bg="#e0e0e0", fg="#000000", font=("Courier New", 14, "bold"), relief=tk.SUNKEN, bd=4, anchor="w", padx=10)
        disp.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 关闭键盘的逻辑
        def close_kbd():
            if hasattr(self, '_current_kbd') and self._current_kbd:
                self._current_kbd.destroy()
                self._current_kbd = None
            
        tk.Button(top_frame, text="RETURN", font=("Courier New", 14, "bold"), bg="#ff9800", fg="#111", 
                  activebackground="#e68a00", bd=3, command=close_kbd).pack(side=tk.RIGHT, padx=(10, 0))

        layout = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'DEL'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'CLR', 'OK'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '_', '.', 'SPACE']
        ]

        def click(btn_val):
            current = target_var.get()
            if btn_val == 'OK': 
                close_kbd()
            elif btn_val == 'CLR': 
                target_var.set("")
            elif btn_val == 'DEL':
                target_var.set(current[:-1]) 
            elif btn_val == 'SPACE':
                target_var.set(current + " ")
            else: 
                target_var.set(current + btn_val)

        # Grid
        keys_frame = tk.Frame(pad, bg="#2b2b2b")
        keys_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=(0, 10))

        for r in range(4): keys_frame.grid_rowconfigure(r, weight=1)
        for c in range(11): keys_frame.grid_columnconfigure(c, weight=1)

        for r_idx, row in enumerate(layout):
            for c_idx, key in enumerate(row):
                color = "#a3a3a3"
                fg_color = "#111"
                
                if key == 'OK': color = "#4caf50"
                elif key == 'CLR' or key == 'DEL': color = "#f44336"
                
                colspan = 2 if key == 'SPACE' else 1
                
                btn = tk.Button(keys_frame, text=key, font=("Courier New", 13, "bold"), bg=color, fg=fg_color, 
                                activebackground="#d4d4d4", command=lambda b=key: click(b))
                btn.grid(row=r_idx, column=c_idx, columnspan=colspan, sticky="nsew", padx=2, pady=2)
                                                
    def show_execute_page(self, event=None): self.execute_page.tkraise()
    def show_config_page(self, event=None): self.config_page.tkraise()
    def show_scan_page(self, event=None): self.scan_page.tkraise()
    def show_task_page(self, event=None): self.task_page.tkraise()
    
    def apply_settings(self):
        if not self.hardware_connected: return
        self.is_configuring = True
        try:
            start_nm = int(self.wave_start_var.get())
            end_nm = int(self.wave_end_var.get())
            if start_nm >= end_nm: raise ValueError("Start wavelength must be < end wavelength!")
            if start_nm < 900 or end_nm > 1700: raise ValueError("Bounds exceeded (900-1700nm)!")
            
            scan_mode_str = self.scan_combo.get()
            scan_type = 1 if scan_mode_str == "Hadamard" else 0
            pga_val = self._parse_pga_gain()
            is_lamp_on = self.lamp_var.get()

            if self.hardware_connected and self.nirs is not None:
                self._force_wake_up_hardware()
                self.nirs.set_lamp_on_off(is_lamp_on)
                time.sleep(1.0)
                self.nirs.set_config(scanConfigIndex=8, scan_type=scan_type, num_patterns=228, num_repeats=6, wavelength_start_nm=start_nm, wavelength_end_nm=end_nm, width_px=7)
                self.nirs.set_pga_gain(pga_val)

            # --- Debug: print current settings ---
            print("[DEBUG] === Settings Applied ===")
            print(f"[DEBUG] Wavelength: {start_nm}-{end_nm} nm")
            print(f"[DEBUG] Scan Mode: {scan_mode_str}, Num Patterns: 228, Num Repeats: 6")
            print(f"[DEBUG] PGA Gain: {self.pga_combo.get()}({pga_val}), Lamp: {'ON' if is_lamp_on else 'OFF'}")
            print(f"[DEBUG] Save Path: {self.save_path_var.get().strip() or 'data/spectra/'}, Save Filename: {self.save_filename_var.get().strip() or '(auto)'}")
            print(f"[DEBUG] Scan Count: {self.scan_count_var.get()}, Scan Interval: {self.scan_interval_var.get()}s")
            print(f"[DEBUG] Hardware Connected: {self.hardware_connected}")

            # 1. 更新 UI 状态指示灯
            self._update_ui_state("CONFIG SAVED", "READY", "#00ff00")
            
            # 2. 🚨 新逻辑：使用弹窗代替执行面板显示
            msg = (f"Configuration Applied Successfully:\n\n"
                   f"Wavelength Range: {start_nm} - {end_nm} nm\n"
                   f"Scan Mode: {scan_mode_str}\n"
                   f"PGA Gain: {self.pga_combo.get()}\n"
                   f"Lamp Status: {'ON' if is_lamp_on else 'OFF'}")
            
            # 使用 messagebox 弹窗确认，并作为主线程的反馈
            messagebox.showinfo("[ CONFIGURATION SUCCESS ]", msg)
            
            # 3. 跳转页面
            self.show_execute_page()
            
            # 移除掉原本对 status_display_var 的 set 调用，保持执行面板清爽
            
        except ValueError as ve:
            messagebox.showerror("[ INPUT ERROR ]", str(ve))
        except Exception as e:
            self.hardware_connected = False
            self._update_ui_state("HW ERROR", "DISCONN", "#ff4444")
            # 硬件错误依然可以弹窗或在执行面板提示
            messagebox.showerror("[ HARDWARE ERROR ]", f"System failure during config:\n{str(e)}")
        finally:
            self.is_configuring = False
   
    def build_config_panel(self):
        center_frame = tk.Frame(self.config_page, bg=self.SCREEN_BG)
        center_frame.place(relx=0.5, rely=0.5, y=12, anchor=tk.CENTER)
        
        tk.Label(center_frame, text="[ SYSTEM CONFIGURATION ]", font=self.FONT_TITLE, bg=self.SCREEN_BG, fg=self.SCREEN_FG).grid(row=0, column=0, columnspan=2, pady=(0, 2))
        
        # The red warning label for configuration page
        tk.Label(center_frame, textvariable=self.config_warn_var, font=("Courier New", 12, "bold"), bg=self.SCREEN_BG, fg="#ff4444").grid(row=1, column=0, columnspan=2, pady=(0, 5))
        
        def screen_label(text): return tk.Label(center_frame, text=text, bg=self.SCREEN_BG, fg=self.SCREEN_FG, font=self.FONT_SYS)

        screen_label("WAVE_START (nm):").grid(row=2, column=0, padx=10, pady=3, sticky=tk.E)
        self.wave_start_var = tk.StringVar(value="900") 
        self.entry_wave_start = tk.Entry(center_frame, textvariable=self.wave_start_var, font=self.FONT_SYS, bg="#b0b0b0", fg=self.SCREEN_FG, relief=tk.SUNKEN, bd=3, width=15)
        
        self.entry_wave_start.config(state="readonly")
        # 改为 ButtonRelease-1
        self.entry_wave_start.bind("<ButtonRelease-1>", lambda e: self.open_keyboard(self.wave_start_var, "WAVE_START"))
        
        self.entry_wave_start.grid(row=2, column=1, padx=10, pady=3, sticky=tk.W)

        screen_label("WAVE_END (nm)  :").grid(row=3, column=0, padx=10, pady=3, sticky=tk.E)
        self.wave_end_var = tk.StringVar(value="1700")
        self.entry_wave_end = tk.Entry(center_frame, textvariable=self.wave_end_var, font=self.FONT_SYS, bg="#b0b0b0", fg=self.SCREEN_FG, relief=tk.SUNKEN, bd=3, width=15)
        
        self.entry_wave_end.config(state="readonly")
        # 改为 ButtonRelease-1
        self.entry_wave_end.bind("<ButtonRelease-1>", lambda e: self.open_keyboard(self.wave_end_var, "WAVE_END"))
        self.entry_wave_end.grid(row=3, column=1, padx=10, pady=3, sticky=tk.W)

        screen_label("PGA_GAIN       :").grid(row=4, column=0, padx=10, pady=3, sticky=tk.E)
        self.pga_combo = ttk.Combobox(center_frame, font=self.FONT_SYS, state="readonly", width=13)
        self.pga_combo["values"] = ("Auto", "1", "2", "4", "8", "16", "32", "64")
        self.pga_combo.current(0)
        self.pga_combo.grid(row=4, column=1, padx=10, pady=3, sticky=tk.W)

        screen_label("SCAN_CONFIG    :").grid(row=5, column=0, padx=10, pady=3, sticky=tk.E)
        self.scan_combo = ttk.Combobox(center_frame, font=self.FONT_SYS, state="readonly", width=13)
        self.scan_combo["values"] = ("Hadamard", "Column")
        self.scan_combo.current(0)
        self.scan_combo.grid(row=5, column=1, padx=10, pady=3, sticky=tk.W)

        screen_label("LAMP_STATUS    :").grid(row=6, column=0, padx=10, pady=3, sticky=tk.E)
        self.lamp_var = tk.BooleanVar(value=False) 

        def toggle_lamp():
            self.lamp_var.set(not self.lamp_var.get())
            if self.lamp_var.get():
                self.lamp_btn.config(text="[ ON ]", bg="#b5b5b5", fg="#1a1a1a")
            else:
                self.lamp_btn.config(text="[ OFF ]", bg="#505050", fg="#ffffff")

        self.lamp_btn = tk.Button(center_frame, text="[ OFF ]", command=toggle_lamp, font=self.FONT_SYS, width=14, bg="#505050", fg="#ffffff", relief=tk.RAISED, bd=3)
        self.lamp_btn.grid(row=6, column=1, sticky=tk.W, padx=10, pady=5)

        screen_label("SAVE_FILENAME  :").grid(row=7, column=0, padx=10, pady=3, sticky=tk.E)
        self.save_filename_var = tk.StringVar(value="")
        self.entry_save_filename = tk.Entry(center_frame, textvariable=self.save_filename_var, font=self.FONT_SYS, bg="#b0b0b0", fg=self.SCREEN_FG, relief=tk.SUNKEN, bd=3, width=15)
        self.entry_save_filename.config(state="readonly")
        self.entry_save_filename.bind("<ButtonRelease-1>", lambda e: self.open_keyboard(self.save_filename_var, "SAVE_FILENAME"))
        self.entry_save_filename.grid(row=7, column=1, padx=10, pady=3, sticky=tk.W)

        screen_label("SAVE_PATH      :").grid(row=8, column=0, padx=10, pady=3, sticky=tk.E)
        self.save_path_var = tk.StringVar(value="data/spectra/")
        self.entry_save_path = tk.Entry(center_frame, textvariable=self.save_path_var, font=self.FONT_SYS, bg="#b0b0b0", fg=self.SCREEN_FG, relief=tk.SUNKEN, bd=3, width=15)
        self.entry_save_path.config(state="readonly")
        self.entry_save_path.bind("<ButtonRelease-1>", lambda e: self.open_keyboard(self.save_path_var, "SAVE_PATH"))
        self.entry_save_path.grid(row=8, column=1, padx=10, pady=3, sticky=tk.W)

        screen_label("SCAN_COUNT     :").grid(row=9, column=0, padx=10, pady=3, sticky=tk.E)
        self.scan_count_var = tk.StringVar(value="1")
        self.entry_scan_count = tk.Entry(center_frame, textvariable=self.scan_count_var, font=self.FONT_SYS, bg="#b0b0b0", fg=self.SCREEN_FG, relief=tk.SUNKEN, bd=3, width=15)
        self.entry_scan_count.config(state="readonly")
        self.entry_scan_count.bind("<ButtonRelease-1>", lambda e: self.open_keyboard(self.scan_count_var, "SCAN_COUNT"))
        self.entry_scan_count.grid(row=9, column=1, padx=10, pady=3, sticky=tk.W)

        screen_label("SCAN_INTERVAL  :").grid(row=10, column=0, padx=10, pady=3, sticky=tk.E)
        self.scan_interval_var = tk.StringVar(value="0")
        self.entry_scan_interval = tk.Entry(center_frame, textvariable=self.scan_interval_var, font=self.FONT_SYS, bg="#b0b0b0", fg=self.SCREEN_FG, relief=tk.SUNKEN, bd=3, width=15)
        self.entry_scan_interval.config(state="readonly")
        self.entry_scan_interval.bind("<ButtonRelease-1>", lambda e: self.open_keyboard(self.scan_interval_var, "SCAN_INTERVAL"))
        self.entry_scan_interval.grid(row=10, column=1, padx=10, pady=3, sticky=tk.W)

        screen_label("HARDWARE RESET :").grid(row=11, column=0, padx=10, pady=3, sticky=tk.E)

        def reset_hardware_fault():
            try:
                if self.hardware_connected and self.nirs is not None:
                    if hasattr(self.nirs, 'resetErrorStatus'): self.nirs.resetErrorStatus()
                    elif hasattr(self.nirs, 'clear_error_status'): self.nirs.clear_error_status()
                    self._update_ui_state("FAULT CLEARED", "READY", "#00ff00")
            except Exception:
                self.hardware_connected = False
                self._update_ui_state("HW ERROR", "DISCONN", "#ff4444")

        self.reset_btn = tk.Button(center_frame, text="[ CLEAR FAULT ]", command=reset_hardware_fault, font=self.FONT_SYS, width=14, bg="#8b0000", fg="#ffffff", activebackground="#ff4444", relief=tk.RAISED, bd=3)
        self.reset_btn.grid(row=11, column=1, sticky=tk.W, padx=10, pady=5)

        btn_frame = tk.Frame(center_frame, bg=self.SCREEN_BG)
        btn_frame.grid(row=12, column=0, columnspan=2, pady=10)
        tk.Button(btn_frame, text="< RETURN", command=self.show_execute_page, font=self.FONT_SYS, bg="#cccccc", fg=self.SCREEN_FG, relief=tk.RAISED, bd=3, width=12).pack(side=tk.LEFT, padx=15)
        self.apply_btn = tk.Button(btn_frame, text="APPLY >", command=self.apply_settings, bg="#a3a3a3", fg=self.SCREEN_FG, font=self.FONT_SYS, relief=tk.RAISED, bd=3, width=12)
        self.apply_btn.pack(side=tk.LEFT, padx=15)

    def build_execute_panel(self):
        # 1. 动态任务名称 (减小顶部留白，适配 5寸屏)
        self.current_task_var = tk.StringVar(value="CURRENT TASK: NONE")
        tk.Label(self.execute_page, textvariable=self.current_task_var, font=self.FONT_TITLE, 
                 bg=self.SCREEN_BG, fg=self.SCREEN_FG).pack(pady=(10, 5))

        # 2. 动态任务简介
        self.task_desc_var = tk.StringVar(value="Select a task to proceed.")
        tk.Label(self.execute_page, textvariable=self.task_desc_var, font=("Courier New", 10), 
                 bg=self.SCREEN_BG, fg="#333333", wraplength=500, justify=tk.LEFT).pack(pady=(0, 5))

        # 警告标签
        tk.Label(self.execute_page, textvariable=self.execute_warn_var, font=("Courier New", 12, "bold"), 
                 bg=self.SCREEN_BG, fg="#ff4444").pack(pady=(0, 5))

        # 3. 中间状态显示框 (🚨修复：改为深色终端风格，缩减高度边距)
        self.result_frame = tk.Frame(self.execute_page, bg="#222222", bd=4, relief=tk.SUNKEN)
        self.result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=25, pady=(5, 15))
        
        self.center_content = tk.Frame(self.result_frame, bg="#222222")
        self.center_content.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.main_display = tk.Label(self.center_content, textvariable=self.status_display_var, 
                                     font=("Courier New", 14, "bold"), bg="#222222", fg="#39ff14", justify=tk.CENTER)
        self.main_display.pack(pady=10)
        
        # --- 4. 底部动作按钮区域 ---
        self.action_btn_frame = tk.Frame(self.execute_page, bg=self.SCREEN_BG)
        self.action_btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=40, pady=(0, 20))

        self.action_btn_frame.columnconfigure(0, weight=1)
        self.action_btn_frame.columnconfigure(1, weight=1)

        # 🚨修复：训练按钮初始默认为灰色禁用状态
        self.train_btn = tk.Button(self.action_btn_frame, text="[ TRAIN ]", command=self.show_train_page, 
                                   font=("Courier New", 16, "bold"), bg="#5a5a5a", fg="#ffffff", 
                                   bd=5, relief=tk.RAISED, height=2, state=tk.DISABLED)
        self.train_btn.grid(row=0, column=0, padx=(0, 10), sticky="ew")

        # 预测按钮初始同样禁用
        self.predict_btn = tk.Button(self.action_btn_frame, text="[ PREDICT ]", command=self.show_predict_page, 
                                     font=("Courier New", 16, "bold"), bg="#5a5a5a", fg="#ffffff", 
                                     activebackground="#5F9EA0", bd=5, relief=tk.RAISED, height=2, 
                                     state=tk.DISABLED) 
        self.predict_btn.grid(row=0, column=1, padx=(10, 0), sticky="ew")
    
    def build_train_panel(self):
        """构建工业级训练界面：左右分栏布局，完美契合 5寸宽屏黄金比例"""
        
        # 1. 顶部标题区 (左右对齐，利用宽度)
        header_frame = tk.Frame(self.train_page, bg=self.SCREEN_BG)
        header_frame.pack(fill=tk.X, pady=(15, 10), padx=20)

        tk.Label(header_frame, text="[ MODEL TRAINING ]", font=("Courier New", 16, "bold"), 
                 bg=self.SCREEN_BG, fg=self.SCREEN_FG).pack(side=tk.LEFT)

        self.train_task_name_var = tk.StringVar(value="TASK: N/A")
        tk.Label(header_frame, textvariable=self.train_task_name_var, font=("Courier New", 12, "bold"), 
                 bg=self.SCREEN_BG, fg="#111111").pack(side=tk.RIGHT)

        # 2. 核心内容区 (采用左右双面板设计，极致美观)
        content_frame = tk.Frame(self.train_page, bg=self.SCREEN_BG)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        # -- 左侧：参数表单区 (占宽屏的左半边) --
        left_col = tk.Frame(content_frame, bg="#b0b0b0", bd=3, relief=tk.SUNKEN)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 将网格放在左侧面板居中
        grid_frame = tk.Frame(left_col, bg="#b0b0b0")
        grid_frame.pack(expand=True) 
        
        def form_label(text): return tk.Label(grid_frame, text=text, bg="#b0b0b0", fg="#111111", font=("Courier New", 11, "bold"))

        # 分类/回归组件预埋 (由 show_train_page 动态 grid 控制)
        self.lbl_train_cls = form_label("TAG:")
        self.train_tag_combo = ttk.Combobox(grid_frame, textvariable=self.train_tag_var, font=self.FONT_SYS, state="readonly", width=14)
        
        self.lbl_train_reg = form_label("VAL:")
        self.entry_train_target = tk.Entry(grid_frame, textvariable=self.train_target_var, font=self.FONT_SYS, bg="#e0e0e0", width=15)
        self.entry_train_target.bind("<ButtonRelease-1>", lambda e: self.open_keyboard(self.train_target_var, "TARGET VALUE"))
        
        self.lbl_range_hint = tk.Label(grid_frame, text="Range: --", font=("Courier New", 9), bg="#b0b0b0", fg="#444")

        # -- 右侧：终端日志区 (占宽屏的右半边) --
        right_col = tk.Frame(content_frame, bg="#222222", bd=4, relief=tk.SUNKEN)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right_col, text="SYSTEM LOG", font=("Courier New", 9, "bold"), bg="#222222", fg="#888888", anchor="w").pack(fill=tk.X, padx=8, pady=(4, 0))
        
        # 日志框现在可以非常舒展地占据右侧高度
        self.train_console = tk.Listbox(right_col, font=("Courier New", 10), bg="#222222", fg="#39ff14", bd=0, highlightthickness=0)
        self.train_console.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # 3. 底部固定按钮区 (左右对齐，拉伸按钮体积)
        btn_box = tk.Frame(self.train_page, bg=self.SCREEN_BG)
        btn_box.pack(fill=tk.X, padx=20, pady=(10, 20))
        
        tk.Button(btn_box, text="< CANCEL", font=("Courier New", 14, "bold"), bg="#cccccc", width=12, bd=4,
                  command=self.show_execute_page).pack(side=tk.LEFT)
        
        self.start_train_btn = tk.Button(btn_box, text="[ TRAIN ]", font=("Courier New", 14, "bold"), 
                                         bg="#e67e22", fg="white", width=14, bd=4,
                                         command=self._on_start_train_clicked)
        self.start_train_btn.pack(side=tk.RIGHT)
        

    def show_train_page(self):
        """完全基于 JSON 的解析逻辑，彻底删除旧版的正则切割"""
        import json # 确保引入了 json
        
        if not self.tasks: return
        
        task = self.tasks[self.current_task_idx]
        task_name = task.get("name", "Unknown Task")  
        task_type = task.get("task_type", "Classification")
        raw_description = task.get("description", "")
        
        # 1. 刷新文本看板
        self.train_task_name_var.set(f"TASK: {task_name.upper()}")
        self.train_task_type_var.set(f"MODE: {task_type.upper()}")
        
        self.train_console.delete(0, tk.END)
        self.train_console.insert(tk.END, "> Pipeline Ready. Awaiting data inputs.")
        # 把数据库里读到的最原始字符串直接打在黑色日志框里！
        self.train_console.insert(tk.END, f"> [DEBUG] DB_RAW: {raw_description}")
        
        # 2. 清理历史网格状态
        self.lbl_train_cls.grid_remove(); self.train_tag_combo.grid_remove()
        self.lbl_train_reg.grid_remove(); self.entry_train_target.grid_remove(); self.lbl_range_hint.grid_remove()
        
        
        try:
            meta_data = json.loads(raw_description)
        except Exception as e:
            # 如果走到这里，终端一定会报错，我们就知道为什么失败了
            print(f"\n[FATAL ERROR] JSON 解析彻底失败！")
            print(f"报错原因: {e}")
            print(f"数据库里存的错误字符串是: {raw_description}\n")
            meta_data = {} # 给个空字典防崩溃
            
        # 4. 根据 JSON 直接赋值，再也没有 split() 了！
        if task_type == "Classification":
            self.lbl_train_cls.grid(row=1, column=0, padx=10, pady=10, sticky=tk.E)
            self.train_tag_combo.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)
            
            # 直接从 JSON 里拿 classes 数组
            tags_list = meta_data.get("classes", ["Error: No Tags Found"])
            self.train_tag_combo["values"] = tags_list
            if tags_list: self.train_tag_combo.current(0)
                
        else:  # Regression 回归任务
            self.lbl_train_reg.grid(row=1, column=0, padx=10, pady=10, sticky=tk.E)
            self.entry_train_target.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)
            self.lbl_range_hint.grid(row=2, column=1, padx=10, pady=(0, 5), sticky=tk.W)
            
            # 直接从 JSON 里拿 reg_range 字符串
            range_str = meta_data.get("reg_range", "")
            self.lbl_range_hint.config(text=f"Expected Range: [{range_str}]")
            self.train_target_var.set("") # 清空输入框

        self.train_page.tkraise()

    def _on_start_train_clicked(self):
        """输入有效性防呆校验拦截器 (支持动态 JSON 边界识别)"""
        task = self.tasks[self.current_task_idx]
        task_type = task.get("task_type", "Classification")
        
        if task_type == "Regression":
            raw_val = self.train_target_var.get().strip()
            if not raw_val:
                messagebox.showerror("Error", "Please input a valid target calibration numerical value!")
                return
            try:
                val = float(raw_val)
                
                # 🚨 核心修复：从 JSON 中动态解析范围，替代丢失的 self.reg_min_bound
                import json
                try:
                    meta = json.loads(task.get("description", "{}"))
                    reg_range_str = meta.get("reg_range", "")
                    
                    if "-" in reg_range_str:
                        min_str, max_str = reg_range_str.split("-")
                        min_bound = float(min_str.strip())
                        max_bound = float(max_str.strip())
                        
                        # 超出安全范围时，弹出选择框，而不是直接报错卡死
                        if val < min_bound or val > max_bound:
                            user_ok = messagebox.askyesno(
                                "Out of Bounds Warning", 
                                f"Warning: The input value [{val}] is outside the expected range [{min_bound} - {max_bound}] defined for this task.\n\nDo you want to forcefully proceed with this sample?"
                            )
                            if not user_ok:
                                return # 用户点击否，放弃训练
                except Exception as e:
                    print(f"[DEBUG] Range validation skipped due to parsing error: {e}")
                    pass # 如果范围没写或者格式不对，就不进行范围拦截，直接放行
                    
            except ValueError:
                messagebox.showerror("Format Error", "Target calibration value must be a valid float/integer!")
                return
                
        # 校验通过，锁定全局按钮，拉起子线程，保障硬件通信时UI不卡死
        self.start_train_btn.config(state=tk.DISABLED)
        self.train_console.delete(0, tk.END)
        self.train_console.insert(tk.END, "> Starting Data Acquisition Pipeline...")
        
        # 核心：告诉底层状态机，接下来的这一枪是用来 train 的！
        self.current_action = "train"
        
        # 调用你已有的底层硬件扫描函数
        if hasattr(self, 'execute_scan'):
            self.execute_scan(action="train")
        else:
            print("[ERROR] 找不到 execute_scan 函数！")
        









            
                
    def build_predict_panel(self):
        """构建在线预测引擎界面 (工业级结果展示仪表盘)"""
        # 1. 顶部状态看板
        tk.Label(self.predict_page, text="[ ONLINE PREDICTION ENGINE ]", font=("Courier New", 14, "bold"), bg=self.SCREEN_BG, fg=self.SCREEN_FG).pack(pady=(15, 5))
        
        # 2. 核心预测仪表盘框架 (采用深色科技感背板，极致压缩高度)
        dashboard_frame = tk.Frame(self.predict_page, bg="#1a1a1a", bd=4, relief=tk.RIDGE)
        dashboard_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=5)
        
        # --- 2.1 任务选择区 ---
        top_ctrl_frame = tk.Frame(dashboard_frame, bg="#1a1a1a")
        top_ctrl_frame.pack(fill=tk.X, pady=10, padx=20)
        
        tk.Label(top_ctrl_frame, text="DEPLOYED MODEL:", font=("Courier New", 10, "bold"), bg="#1a1a1a", fg="#00bcd4").pack(side=tk.LEFT)
        
        self.predict_task_var = tk.StringVar()
        self.predict_task_combo = ttk.Combobox(top_ctrl_frame, textvariable=self.predict_task_var, font=self.FONT_SYS, state="readonly", width=25)
        self.predict_task_combo.pack(side=tk.LEFT, padx=10)
        self.predict_task_combo.bind("<<ComboboxSelected>>", self._on_predict_task_selected)

        # --- 2.2 巨型结果显示器 ---
        self.lbl_big_result = tk.Label(dashboard_frame, textvariable=self.predict_result_var, 
                                       font=("Courier New", 20, "bold"), bg="#051005", fg="#39ff14", 
                                       width=24, height=2, relief=tk.SUNKEN, bd=4)
        self.lbl_big_result.pack(pady=5)

        # --- 2.3 模型信息条 ---
        self.predict_model_info_var = tk.StringVar(value="-- No model selected --")
        tk.Label(dashboard_frame, textvariable=self.predict_model_info_var, font=("Courier New", 11), bg="#1a1a1a", fg="#00bcd4").pack(pady=(0, 2))

        # --- 2.4 置信度/补充信息条 (高亮醒目版) ---
        self.predict_detail_var = tk.StringVar(value="-- Waiting for execution --")
        tk.Label(dashboard_frame, textvariable=self.predict_detail_var, font=("Courier New", 14, "bold"), bg="#1a1a1a", fg="#ffeb3b").pack(pady=5)

        # --- 2.5 模型性能展示区 (批量预测时显示准确度+混淆矩阵) ---
        self.predict_perf_var = tk.StringVar(value="")
        tk.Label(dashboard_frame, textvariable=self.predict_perf_var, font=("Courier New", 9),
                 bg="#1a1a1a", fg="#aaaaaa", justify=tk.LEFT, anchor=tk.W).pack(pady=(5, 0), padx=20)

        # 3. 底部控制按钮区域
        bottom_frame = tk.Frame(self.predict_page, bg=self.SCREEN_BG)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 15), padx=30)
        
        tk.Button(bottom_frame, text="< CANCEL", font=("Courier New", 14, "bold"), bg="#cccccc", fg="#111111", width=12, bd=4,
                  command=self.show_execute_page).pack(side=tk.LEFT)
                  
        self.start_predict_btn = tk.Button(bottom_frame, text="[ PREDICT ]", font=("Courier New", 14, "bold"), 
                                           bg="#00bcd4", fg="#ffffff", activebackground="#0097a7", width=12, bd=4,
                                           command=self._on_start_predict_clicked)
        self.start_predict_btn.pack(side=tk.RIGHT)
        
    def show_predict_page(self):
        """进入预测页面时的初始化：强制同步数据 + 筛选可用模型"""
        import json
        
        # 🚨 核心修复：进入页面瞬间，强制刷新一次数据库列表！
        # 这确保了你在"训练页面"保存的 is_trained=1 状态能立刻同步到这里
        if hasattr(self, '_load_tasks_from_db'):
            self._load_tasks_from_db()

        self.predict_result_var.set("READY.\nAWAITING SCAN.")
        self.predict_detail_var.set("--")
        self.predict_perf_var.set("")
        
        # 过滤出所有已经训练过的可用模型
        self.available_predict_tasks = [t for t in getattr(self, 'tasks', []) if t.get("is_trained") == 1]
        
        if not self.available_predict_tasks:
            self.predict_task_combo["values"] = ["No trained models available!"]
            self.predict_task_combo.set("No trained models available!") # 用 set 替换 current(0) 更稳健
            self.start_predict_btn.config(state=tk.DISABLED)
        else:
            self.start_predict_btn.config(state=tk.NORMAL)
            # 把任务名称放进下拉列表
            self.predict_task_combo["values"] = [t["name"] for t in self.available_predict_tasks]
            self.predict_task_combo.current(0)
            
            # 手动触发一次，更新描述信息（传入None作为event参数）
            self._on_predict_task_selected(None) 
            
        self.predict_page.tkraise()

    def _on_predict_task_selected(self, event):
        """用户在下拉菜单切换模型时，解析 JSON 并展示模型参数"""
        import json
        if not self.available_predict_tasks: return
        
        idx = self.predict_task_combo.current()
        if idx < 0: return
        
        selected_task = self.available_predict_tasks[idx]
        # 把这个选中的任务设为当前活动的任务，以便底层的 predict 逻辑调用
        self.current_task_idx = self.tasks.index(selected_task) 
        
        try:
            meta = json.loads(selected_task.get("description", "{}"))
            t_type = meta.get("task_type", "Unknown")
            algo = meta.get("algorithm", "Unknown")
            info_text = f"Type: {t_type} | Algo: {algo}"
            self.predict_model_info_var.set(info_text)
        except:
            self.predict_model_info_var.set("Meta data parsing failed.")

    def _on_start_predict_clicked(self):
        """点击预测按钮：锁定 UI，改变状态机，触发硬件扫描"""
        self.start_predict_btn.config(state=tk.DISABLED)
        self.predict_result_var.set("SCANNING...\nPLEASE WAIT")
        
        # 核心：告诉底层状态机，接下来的这一枪是用来 predict 的！
        self.current_action = "predict" 
        
        # 调用你已有的底层硬件扫描函数
        if hasattr(self, 'execute_scan'):
            self.execute_scan(action="predict")
        else:
            print("[ERROR] 找不到 execute_scan 函数！")
        
    def build_scan_panel(self):
    
        # Top title
        tk.Label(self.scan_page, text="[ SPECTRAL OSCILLOSCOPE ]", font=self.FONT_TITLE, bg=self.SCREEN_BG, fg=self.SCREEN_FG).pack(side=tk.TOP, pady=(15, 5))
        
        # The red warning label for scan page
        tk.Label(self.scan_page, textvariable=self.scan_warn_var, font=("Courier New", 12, "bold"), bg=self.SCREEN_BG, fg="#ff4444").pack(side=tk.TOP, pady=(0, 5))
        
        
        # 2. Buttom Button
        btn_frame = tk.Frame(self.scan_page, bg=self.SCREEN_BG)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(0, 20))
        tk.Button(btn_frame, text="< SETTINGS", command=self.show_config_page, font=self.FONT_SYS, bg="#cccccc", fg=self.SCREEN_FG, relief=tk.RAISED, bd=3, width=12).pack(side=tk.LEFT)
        
        self.scan_scan_btn = tk.Button(btn_frame, text="[ EXECUTE SCAN ]", command = partial(self.execute_scan, action="scan",show_progress=False), font=("Courier New", 14, "bold"), bg="#a3a3a3", fg="#1a1a1a", relief=tk.RAISED, bd=4)
        self.scan_scan_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(15, 0))

        mode_frame = tk.Frame(self.scan_page, bg=self.SCREEN_BG)
        mode_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(5, 10))
        mode_frame.columnconfigure(0, weight=1); mode_frame.columnconfigure(1, weight=1); mode_frame.columnconfigure(2, weight=1)

        self.graph_mode = 'intensity' 
        with self.scan_data_lock:
            self.scan_data = {'wave': [], 'intensity': [], 'reflectance': [], 'absorbance': []}

        def create_tab_btn(parent, text, mode, col):
            btn = tk.Button(parent, text=text, font=("Courier New", 11, "bold"), command=lambda: self.set_graph_mode(mode), relief=tk.RAISED, bd=3)
            btn.grid(row=0, column=col, padx=5, sticky="ew")
            return btn

        self.btn_int = create_tab_btn(mode_frame, "[ INTENSITY ]", 'intensity', 0)
        self.btn_abs = create_tab_btn(mode_frame, "[ ABSORBANCE ]", 'absorbance', 1)
        self.btn_ref = create_tab_btn(mode_frame, "[ REFLECTANCE ]", 'reflectance', 2)

        # Middle Canvas
        self.scan_frame = tk.Frame(self.scan_page, bg="#505050", bd=6, relief=tk.SUNKEN)
        self.scan_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        self.scan_canvas = tk.Canvas(self.scan_frame, bg="#051005", highlightthickness=0)
        self.scan_canvas.pack(fill=tk.BOTH, expand=True)
        self.scan_canvas.bind("<Configure>", lambda e: self._draw_graph())

        self.set_graph_mode('intensity')
        



    def build_left_panel(self):
        tk.Label(self.ctrl_frame, text="⊕", bg=self.PANEL_BG, fg="#4d4d4d", font=("Arial", 14)).place(x=5, y=5)
        tk.Label(self.ctrl_frame, text="⊕", bg=self.PANEL_BG, fg="#4d4d4d", font=("Arial", 14)).place(relx=1.0, x=-25, y=5)
          
        self.status_label = tk.Label(self.ctrl_frame, text="SYS: BOOTING...\nHW : [CHECKING]\nIP : LOADING...", font=("Courier New", 11, "bold"), bg="#8a8a8a", fg="#111111", bd=3, relief=tk.SUNKEN, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X, padx=15, pady=(35, 5))

        self.status_frame = tk.Frame(self.ctrl_frame, bg=self.PANEL_BG)
        self.status_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 10))

        # 2. 将电池标签放进状态栏，靠左对齐
        # (提示: 为了能和温度挤在同一行，我把电池的字体从 18 稍微调成了 14)
        self.lbl_battery = tk.Label(self.status_frame, text="BAT:--%", font=("Courier New", 12, "bold"), bg=self.PANEL_BG, fg="white")
        self.lbl_battery.pack(side=tk.LEFT, anchor="w")

        # 3. 将温度标签放进状态栏，靠右对齐
        self.temp_label = tk.Label(self.status_frame, text="TEMP:N/A", font=("Courier New", 12, "bold"), bg="#8a8a8a", fg="#111111", bd=3, relief=tk.SUNKEN)
        self.temp_label.pack(side=tk.RIGHT, anchor="e")

        def create_mech_button(text, command, is_dark=False):
            return tk.Button(self.ctrl_frame, text=text, command=command, font=self.FONT_SYS, bg="#666666" if is_dark else self.BTN_BG, fg="#ffffff" if is_dark else "#111111", activebackground="#8c8c85", bd=5, relief=tk.RAISED, height=2)
              
        tk.Button(self.ctrl_frame, text="[ SCAN ]", command=self.show_scan_page, font=self.FONT_SYS, bg=self.BTN_BG, fg="#111111", activebackground="#8c8c85", bd=5, relief=tk.RAISED, height=2).pack(fill=tk.X, padx=15, pady=4)
        tk.Button(self.ctrl_frame, text="[ SETTINGS ]", command=self.show_config_page, font=self.FONT_SYS, bg=self.BTN_BG, fg="#111111", activebackground="#8c8c85", bd=5, relief=tk.RAISED, height=2).pack(fill=tk.X, padx=15, pady=4)
        tk.Button(self.ctrl_frame, text="[ EXECUTE ]", command=self.show_execute_page, font=self.FONT_SYS, bg=self.BTN_BG, fg="#111111", activebackground="#8c8c85", bd=5, relief=tk.RAISED, height=2).pack(fill=tk.X, padx=15, pady=4)
        tk.Button(self.ctrl_frame, text="[ TASK LIST ]", command=self.show_task_page, font=self.FONT_SYS, bg=self.BTN_BG, fg="#111111", activebackground="#8c8c85", bd=5, relief=tk.RAISED, height=2).pack(fill=tk.X, padx=15, pady=4)
        
        tk.Frame(self.ctrl_frame, height=2, bg="#5c5e5c", bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=10, pady=5)
        self.exit_btn = create_mech_button("[ EXIT ]", self.root.quit, is_dark=True)
        self.exit_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=10)
        
        
    def execute_scan(self, action="scan", show_progress=True): 
        """
        统一的硬件扫描发令枪
        :param action: "scan", "predict", "train"
        :param show_progress: 是否显示进度条 (默认开启)
        """
        if not self.hardware_connected: return
        self.is_scanning = True
        self._scan_gen += 1   # bump generation for cleanup race guard
        self.current_action = action 
        
        self._update_ui_state("SCANNING", "SCANNING", "#ffaa00")

        
       
            
        if hasattr(self, 'status_display_var'):
            self.status_display_var.set("INITIATING SCAN SEQUENCE...")
            
        self.scan_data = {'wave': [], 'intensity': [], 'reflectance': [], 'absorbance': []}
        
        if hasattr(self, 'scan_canvas') and self.scan_canvas.winfo_exists():
            self.scan_canvas.delete("all")
            self.scan_canvas.create_text(self.scan_canvas.winfo_width()/2, self.scan_canvas.winfo_height()/2, 
                                        text="[ SCAN IN PROGRESS... ]", fill="#ffaa00", font=("Courier New", 14, "bold"))
            
        threading.Thread(target=self._hardware_scan_task, daemon=True).start()
        
    # Task Panel Components (With Live Swipe Animation)

        










        
        
    

        




            
        
        


    
            


        


    def _update_battery_status(self):
        """后台非阻塞的电池状态轮询引擎 (含 INA219 库仑计数)"""
        if self.power_sensor:
            try:
                bus_voltage = self.power_sensor.getBusVoltage_V()
                current_mA = self.power_sensor.getCurrent_mA()

                # Coulomb counter: integrate current over time (mAh)
                now = time.time()
                if hasattr(self, '_battery_last_ts'):
                    dt_hours = (now - self._battery_last_ts) / 3600.0
                    self._battery_consumed_mah += abs(current_mA) * dt_hours
                self._battery_last_ts = now

                # Coulomb-counted SOC (primary metric)
                coulomb_pct = max(0, 100 - (self._battery_consumed_mah / self._battery_capacity_mah) * 100)
                # Voltage-based SOC (fallback, linear approximation for 2S LiPo)
                voltage_pct = (bus_voltage - 6) / 2.4 * 100

                # Blend: prefer coulomb counting once we have meaningful data (>0.5 mAh consumed)
                if self._battery_consumed_mah > 0.5:
                    p = max(0, min(100, int(coulomb_pct)))
                else:
                    p = max(0, min(100, int(voltage_pct)))

                # 根据电量决定颜色和报警逻辑
                if p > 50:
                    color = "#39ff14" # 绿色：健康
                    self.battery_warned = False # 充上电后重置警告标志
                elif p > 20:
                    color = "#ffeb3b" # 黄色：中等
                    self.battery_warned = False
                else:
                    color = "#ff4444" # 红色：危险
                    if not self.battery_warned:
                        self._show_low_battery_warning(p, bus_voltage)
                        self.battery_warned = True # 标记已警告，直到电量恢复前不再弹

                # 更新左侧面板的 UI
                self.lbl_battery.config(text=f"BAT: {p}%", fg=color)

            except Exception as e:
                self.lbl_battery.config(text="BAT: ERR", fg="#ff4444")
                print(f"[HW_ERR] 读取电量失败: {e}")

        # 每 5 秒钟自动调用一次自己，形成后台轮询，绝对不卡死 UI
        self.root.after(5000, self._update_battery_status)


    def _show_low_battery_warning(self, percent, voltage):
        """极其暴力的全屏红色警告弹窗 (绝对防遮挡)"""
        # 使用内嵌图层法，直接盖在所有东西的最上面
        warn_overlay = tk.Frame(self.root, bg="#ff0000", bd=10, relief=tk.RAISED)
        warn_overlay.place(relx=0.5, rely=0.5, relwidth=0.6, relheight=0.4, anchor=tk.CENTER)
        warn_overlay.tkraise()

        # 警告图标/文字
        tk.Label(warn_overlay, text="⚠️ CRITICAL WARNING", font=("Courier New", 18, "bold"), bg="#ff0000", fg="white").pack(pady=(20, 10))
        tk.Label(warn_overlay, text=f"LOW BATTERY: {percent}%\nVOLTAGE: {voltage:.2f}V", font=("Courier New", 16, "bold"), bg="#ff0000", fg="white").pack(pady=10)
        tk.Label(warn_overlay, text="Please connect to power immediately!", font=("Courier New", 12), bg="#ff0000", fg="white").pack(pady=(0, 15))

        # 确认按钮（点击销毁这个图层）
        tk.Button(warn_overlay, text="ACKNOWLEDGE", font=("Courier New", 16, "bold"), bg="white", fg="#ff0000", 
                  command=warn_overlay.destroy, activebackground="#e0e0e0").pack(pady=10)
                  
if __name__ == "__main__":
    try:
        print("[SYS] Starting GUI application...") # 加个日志确认运行到这里
        root = tk.Tk()
        app = SpectrometerApp(root)
        print("[SYS] Main loop starting...")
        root.mainloop() # 确保这一行被执行了
        print("[SYS] Main loop exited. Shutting down...")
    except Exception as e:
        # 这一段能抓到程序为什么闪退的真相
        print(f"!!! 程序即将崩溃，原因如下: {e}")
        import traceback
        traceback.print_exc()
        # 强制让程序暂停，不要直接退出，这样屏幕就不会黑掉
        input("Press Enter to close the window...")
