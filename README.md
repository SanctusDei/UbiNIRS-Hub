# UbiNIRS-Hub

**A self-contained handheld NIR sensing platform — acquisition, preprocessing,
inference, and field adaptation run entirely on-device. No cloud, no tethering.**

Built on Raspberry Pi 3A+ with a TI DLP NIRScan Nano and 5-inch DSI touchscreen.
<br><sub>Jiahao Gong · Xurui Li · Weiwei Jiang</sub>

---

## Hardware

| Component | Notes |
|-----------|-------|
| Spectrometer | TI DLP NIRScan Nano (900–1700 nm, USB HID) |
| Compute | Raspberry Pi 3A+ (BCM2837B0, ARM Cortex-A53) |
| Display | 5-inch DSI touchscreen (800×480) |
| Battery monitor | INA219 over I²C (addr 0x42) — optional |
| Enclosure | Custom 3D-printed housing (STL models in `stl/`) |
| Power | 2S LiPo (7.4 V nominal) — optional |

---

## Quick Start

### 1. Dependencies

```bash
sudo apt-get install libudev-dev libusb-1.0-0-dev
pip install numpy scipy pandas scikit-learn joblib
```

### 2. Compile the native library (Raspberry Pi only)

```bash
chmod +x rebuild_so.sh
./rebuild_so.sh
```

This regenerates the SWIG wrapper from `src/NIRScanner.i`, compiles the C++
sources under `src/`, and links the final `_NIRScanner.so`.

### 3. Launch

```bash
sudo python3 gui_app.py
```

> **Root required** — the NIRScan Nano uses USB HID. See
> [pyusb permissions](https://stackoverflow.com/questions/3738173).

### 4. Boot-time auto-start

```bash
chmod +x setup_autostart.sh
./setup_autostart.sh
```

Configures a systemd `.desktop` entry and passwordless sudo so the GUI
launches automatically on reboot.

---

## Project Structure

```
NIRScanner-Python/
├── gui_app.py             Main GUI (SpectrometerApp) — entry point
├── gui_scan.py            Scan workflow mixin (acquisition, graph, CSV save)
├── gui_ml.py              ML engine mixin (training, inference, memory)
├── gui_tasks.py           Task management mixin (CRUD, card-based browser)
├── NIRS.py                Python wrapper for _NIRScanner.so
├── INA219.py              INA219 battery coulomb counter
├── rebuild_so.sh          Recompile native library from C++ source
├── setup_autostart.sh     Deploy auto-start on boot
│
├── src/                   C++ source & SWIG interface
│   ├── NIRScanner.i       SWIG definition
│   ├── NIRScanner.cpp/h   Python–C++ bridge
│   ├── API.cpp/h          High-level scan API
│   ├── dlpspec_*.c/h      TI DLP Spectrum Library
│   ├── tpl.c/h            TivaWare Peripheral Library
│   ├── hid.c/hidapi.h     USB HID transport
│   ├── evm.cpp/h          EVM (evaluation module) driver
│   ├── usb.cpp/h          USB abstraction
│   ├── serial.c/h         Serial port helpers
│   └── CMakeLists.txt     Build configuration
│
├── stl/                  3D-printed enclosure models
│   ├── Base_Pedestal.STL
│   ├── Base_Back_Cover.STL
│   ├── Spectrometer_Container.STL
│   ├── Lens_Cover.STL
│   └── 3D_Printed_Experiment_Sheet.STL
│
├── models/                Trained model pickles
│
└── data/
    └── spectra/           spectrum CSVs
```

---

## Architecture

The GUI uses a **mixin-based** design on top of Tkinter:

```
┌──────────────────────────────────────────────┐
│              SpectrometerApp                  │
│  gui_app.py — window, hardware state,        │
│  battery heartbeat, page navigation          │
├──────────────────────────────────────────────┤
│  ScanWorkflowMixin      gui_scan.py          │
│  MLEngineMixin           gui_ml.py           │
│  TaskManagerMixin        gui_tasks.py         │
├──────────────────────────────────────────────┤
│  NIRS.py   ←  _NIRScanner.so  ←  C++ SWIG   │
│  INA219.py  (I²C battery sensor)             │
│  SQLite     (spectral_tasks.db)              │
└──────────────────────────────────────────────┘
```

### Key Design Decisions

| Concept | Implementation |
|---------|---------------|
| **Hardware heartbeat** | Background poller thread; auto-disables UI on disconnect, restores on reconnect |
| **Scan generation counter** | Prevents stale scan callbacks from overwriting results after rapid re-scans |
| **Preprocessing pipeline** | SNV → Savitzky-Golay smoothing → derivative → StandardScaler (per-algorithm config) |
| **Stratified memory** | ≤100 samples; ≥3 per class; proportional pruning prevents catastrophic forgetting |
| **Hierarchical classifier** | Two-level HIGH/LOW split on raw absorbance mean; per-branch SVC/KNN/RF |
| **Dual-gate rejection** | Classifier confidence > θ ∧ PCA reconstruction error < τ → accept; else → `UNKNOWN` |
| **Dead-sensor detection** | Rejects scans with peak-to-peak intensity < 1e-6 |

---

## Features

### Spectral Scanning
- Hadamard & Column scan modes over configurable 900–1700 nm range
- Real-time intensity / absorbance / reflectance graph display
- PGA gain, lamp on/off, and hibernation control
- SNR diagnostic scans
- Auto-save as timestamped CSV (`data/spectra/`)

### On-Device Machine Learning
- **Classifiers**: SVM, Random Forest, KNN, LDA
- **Regressors**: SVR, PLS
- **Hierarchical classification** with per-branch confidence estimation
- **Dual-gate sample admission**: confidence threshold + PCA reconstruction-error gate
- **Stratified replay memory**: bounded buffer with class-balanced pruning
- **On-device field adaptation**: incremental retraining without cloud connectivity
- Leave-One-Out cross-validation

### Task Management
- SQLite-backed task CRUD with card-based touch browser
- Classification & regression task types with tag-based organization
- Model export/import (joblib `.pkl`)
- Batch prediction with majority voting

### System
- INA219 battery coulomb counter with capacity estimation
- System IP display with async refresh
- Automatic hardware reconnection on USB disconnect
- Fullscreen 800×480 DSI display with touch keyboard

---

## Usage

### Scanning
1. Open the **SCAN** page
2. Press the large **SCAN** button
3. Toggle **INT / ABS / REF** to switch graph modes
4. Scans auto-save to `data/spectra/`

### Training
1. In **TASKS** → **+ NEW**, choose Classification or Regression
2. Select an algorithm (SVM, RF, KNN, LDA, SVR, PLS)
3. Scan reference samples, assign labels in the training UI
4. Tap **TRAIN** — metrics display automatically
5. The model saves to `models/` and is ready for prediction

### Prediction
1. From the **TASKS** page, select a trained task
2. Switch to **PREDICT** mode
3. Scan an unknown sample — the result appears with confidence score
4. Low-confidence or out-of-distribution scans are rejected as `UNKNOWN` via the dual-gate mechanism

### Battery
When an INA219 sensor is connected, the status bar shows real-time voltage,
current, cumulative mAh consumed, and estimated remaining capacity.

---

## Citation

```bibtex
@inproceedings{gong2026ubinirshub,
  author    = {Gong, Jiahao and Li, Xurui and Jiang, Weiwei},
  title     = {UbiNIRS-Hub: A Mobile Near-Infrared Sensing Platform for
               On-Device Field Adaptation in Ubiquitous Material Analysis},
  year      = {2026},
}
```

---

## Related Repositories

- [NIRScanner-Imaging](https://github.com/HighTemplar-wjiang/NIRScanner-Imaging) — no-reference wavelength selection for NIR imaging
- [NIRScanner-Plotter](https://github.com/HighTemplar-wjiang/NIRScanner-Plotter) — Django server controlling NIRScan Nano + 2D plotter

---

## License

Includes source from the TI DLP NIRscan Nano GUI and DLP Spectrum Library.
Refer to the original distributions for their license terms.
