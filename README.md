# UbiNIRS-Hub — NIRScanner Python Terminal

A full-featured touchscreen GUI application for the **TI DLP NIRScan Nano**
near-infrared spectrometer. Built for Raspberry Pi 3A+ with a 5-inch DSI display,
it combines hardware control, spectral visualization, and on-device machine
learning for real-time material classification and regression with field
adaptation capabilities.

<p align="center">
  <i><b>UbiNIRS-Hub: A Mobile Near-Infrared Sensing Platform for On-Device
  Field Adaptation in Ubiquitous Material Analysis</b></i><br>
  ACM UbiComp/ISWC 2026 &nbsp;|&nbsp;
  <a href="https://github.com/SanctusDei/UbiNIRS-Hub">GitHub</a>
</p>

---

## Features

### 🔬 Spectral Scanning
- **Hadamard & Column** scan modes with configurable wavelength range (900–1700 nm)
- Real-time **intensity / absorbance / reflectance** graph display
- PGA gain, lamp on/off, and hibernation control
- SNR (Signal-to-Noise Ratio) diagnostic scans
- Auto-save spectra as timestamped CSV files

### 🤖 On-Device Machine Learning
- **Classification**: SVM, Random Forest, KNN, LDA
- **Regression**: SVR, PLS
- **Hierarchical classification** with per-branch confidence estimation
- Spectral preprocessing: SNV, MSC, Savitzky-Golay smoothing, absorbance transform
- **StandardScaler** pipeline persistence for KNN/SVM/LDA/SVR
- **Stratified memory management** — bounded replay buffer (max 100 samples, min 3/class)
- **Dual-gate rejection** (Section 2.3.2): classifier confidence + PCA reconstruction-error gate → UNKNOWN rejection
- **On-device field adaptation** — incremental retraining without catastrophic forgetting
- Leave-One-Out cross-validation

### 📋 Task Management
- SQLite-backed task CRUD with **card-based touch browser**
- Classification & regression task types
- Tag-based organization
- Model export/import (joblib `.pkl`)
- Batch prediction with majority voting

### 🔋 System
- **INA219** battery coulomb counter with capacity estimation
- System IP display with async refresh
- Hardware heartbeat watchdog with automatic reconnection
- **Dead sensor detection** — rejects scans with constant/flat intensity
- Auto-start deployment via `setup_autostart.sh`

---

## Hardware Requirements

| Component | Notes |
|-----------|-------|
| **TI DLP NIRScan Nano** | USB HID interface |
| **Raspberry Pi 3A+** | ARM Cortex-A53, Broadcom BCM2837B0 |
| **5-inch DSI Touchscreen** | 800×480 resolution, fullscreen UI |
| **INA219 Current Sensor** (optional) | I²C addr 0x42 for battery monitoring |
| **2S LiPo Battery** (optional) | ~1800 mAh rated capacity |

---

## Software Dependencies

### System

```bash
sudo apt-get install libudev-dev libusb-1.0-0-dev
```

### Python 3

```
numpy>=1.24
scipy>=1.10
pandas>=2.0
scikit-learn>=1.3
joblib>=1.3
tkinter        # included with Python on most systems
```

Install with:

```bash
pip install numpy scipy pandas scikit-learn joblib
```

> **Note**: The NIRScan Nano requires **root/sudo** for USB HID access.
> See [pyusb/libusb permissions](https://stackoverflow.com/questions/3738173/why-does-pyusb-libusb-require-root-sudo-permissions-on-linux).

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SanctusDei/UbiNIRS-Hub.git
cd UbiNIRS-Hub
```

### 2. Compile the Native Library (Raspberry Pi only)

If `_NIRScanner.so` is not present or you need to rebuild:

```bash
chmod +x rebuild_so.sh
./rebuild_so.sh
```

This recompiles the C++ sources under `src/`, regenerates the SWIG wrapper,
and links the final `_NIRScanner.so` shared object.

### 3. Launch the GUI

```bash
sudo python3 main.py
# or directly:
sudo python3 gui_app.py
```

### 4. Auto-Start on Boot

```bash
chmod +x setup_autostart.sh
./setup_autostart.sh
```

This configures a systemd `.desktop` autostart entry and passwordless sudo
for the NIRScanner GUI — the app launches automatically on reboot.

---

## Project Structure

```
UbiNIRS-Hub/
├── main.py                  # Entry point (lightweight launcher)
├── gui_app.py               # Main GUI application (SpectrometerApp)
├── gui_scan.py              # Scan workflow mixin (hardware + graph)
├── gui_ml.py                # ML engine mixin (training + inference)
├── gui_tasks.py             # Task management mixin (CRUD + UI)
├── NIRS.py                  # Python wrapper for _NIRScanner.so
├── INA219.py                # INA219 battery sensor driver
├── setup_autostart.sh       # Raspberry Pi auto-start deployment
├── rebuild_so.sh            # Recompile _NIRScanner.so from C++ source
├── spectral_tasks.db        # SQLite task database (auto-created)
│
├── src/                     # C++ source (SWIG wrapper + DLP spectrum lib)
│   ├── NIRScanner.i         # SWIG interface file
│   ├── NIRScanner.cpp/h     # Python-C++ bridge
│   ├── API.cpp/h            # High-level scan API
│   ├── dlpspec_*.c/h        # DLP Spectrum Library (TI)
│   ├── hid.c/hidapi.h       # USB HID transport
│   ├── tpl.c/h              # TivaWare Peripheral Library
│   └── scripts/             # Compile scripts for py2/py3
│
├── models/                  # Trained ML model pickles
├── dash/                    # Dataset CSVs, cached .npy features, perf logs
├── data/                    # Additional datasets & spectra exports
│   └── spectra/             # Timestamped spectrum CSV archives
│
└── test_/                   # Test scripts & notebooks
```

---

## Architecture

The application uses a **Mixin-based** architecture on top of Tkinter:

```
┌─────────────────────────────────────────────┐
│              SpectrometerApp                 │
│  (gui_app.py — main window, HW state)       │
├─────────────────────────────────────────────┤
│  ScanWorkflowMixin        (gui_scan.py)     │
│  MLEngineMixin            (gui_ml.py)       │
│  TaskManagerMixin         (gui_tasks.py)    │
├─────────────────────────────────────────────┤
│  NIRS.py  ←  _NIRScanner.so  ←  C++ driver │
│  INA219.py (battery sensor)                 │
│  SQLite (spectral_tasks.db)                 │
└─────────────────────────────────────────────┘
```

### Key Design Decisions

- **Hardware heartbeat thread**: Continuously polls the NIRScan device; automatically
  disables UI controls on disconnect and re-enables on reconnect.
- **Scan generation counter**: Prevents stale scan callbacks from overwriting current
  results after rapid re-scans.
- **Stratified sample memory**: Bounded replay buffer (max 100 samples, min 3 per
  class) with proportional pruning — prevents memory domination by recently observed
  classes and mitigates catastrophic forgetting (Section 2.3.1).
- **Hierarchical classification**: Two-level (HIGH/LOW) split based on raw absorbance
  mean, with per-branch SVC/RF/KNN classifiers and per-sample confidence scores
  (Section 2.3.2).
- **Dual-gate rejection mechanism**: Samples are accepted only when classifier
  confidence exceeds the threshold AND PCA reconstruction error stays below the
  learned upper bound — addresses the failure mode where classifiers confidently
  mislabel distribution-shift samples in open-set NIR sensing (Section 2.3.2).
- **On-device field adaptation**: Lightweight retraining pipeline allows the system
  to incorporate novel materials without cloud connectivity or catastrophic forgetting.
- **Dead sensor detection**: Rejects scans where peak-to-peak intensity < 1e-6
  (constant signal across all wavelengths).

---

## Usage

### Scanning

1. Tap **SCAN** to open the scan page
2. Press the large **SCAN** button to acquire a spectrum
3. Use **INT / ABS / REF** buttons to switch graph modes
4. Scans auto-save to `data/spectra/` as CSV

### Training a Model

1. Create a task in **TASKS** → **+ NEW**
2. Choose **Classification** or **Regression**
3. Select an algorithm (SVM, Random Forest, KNN, LDA, SVR, PLS)
4. Scan reference samples and assign labels via the training UI
5. Tap **TRAIN** — performance metrics display automatically
6. The model is saved to `models/` and available for prediction

### Running Predictions

1. From the **TASKS** page, select a trained task
2. Tap **PREDICT** mode
3. Scan an unknown sample
4. The predicted class/value appears with confidence score
5. Low-confidence predictions are rejected as **UNKNOWN** via the dual-gate
   mechanism (classifier confidence + PCA reconstruction error)

### Battery Monitoring

When an INA219 sensor is connected, the status bar shows:
- Instantaneous voltage & current
- Cumulative mAh consumed (coulomb counting)
- Estimated remaining capacity

---

## Citation

If you use this software in your research, please cite:

```bibtex
@inproceedings{gong2026ubinirshub,
  author = {Gong, Jiahao and Li, Xurui and Jiang, Weiwei},
  title = {UbiNIRS-Hub: A Mobile Near-Infrared Sensing Platform for
           On-Device Field Adaptation in Ubiquitous Material Analysis},
  booktitle = {Proc. UbiComp/ISWC '26},
  year = {2026},
  publisher = {ACM},
  address = {Shanghai, China},
  doi = {10.1145/XXXXXXX.XXXXXXX}
}

@article{jiang2020probing,
  author = {Jiang, Weiwei and Marini, Gabriele and van Berkel, Niels
            and Sarsenbayeva, Zhanna and Tan, Zheyu and Luo, Chu and
            He, Xin and Dingler, Tilman and Goncalves, Jorge and
            Kawahara, Yoshihiro and Kostakos, Vassilis},
  title = {Probing Sucrose Contents in Everyday Drinks Using
           Miniaturized Near-Infrared Spectroscopy Scanners},
  journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  volume = {3},
  number = {4},
  year = {2020},
  doi = {10.1145/3369834}
}

@article{jiang2022near,
  author = {Jiang, Weiwei and Yu, Difeng and Wang, Chaofan and
            Sarsenbayeva, Zhanna and van Berkel, Niels and
            Goncalves, Jorge and Kostakos, Vassilis},
  title = {Near-infrared Imaging for Information Embedding and
           Extraction with Layered Structures},
  journal = {ACM Trans. Graph.},
  volume = {42},
  number = {1},
  year = {2022},
  doi = {10.1145/3533426}
}
```

---

## Related Repositories

- [NIRScanner-Imaging](https://github.com/HighTemplar-wjiang/NIRScanner-Imaging) — No-reference wavelength selection algorithm for near-infrared imaging
- [NIRScanner-Plotter](https://github.com/HighTemplar-wjiang/NIRScanner-Plotter) — Django server controlling NIRScan Nano + 2D plotter

---

## License

This repository incorporates source code from the DLP NIRscan Nano GUI and
DLP Spectrum Library (Texas Instruments). Please refer to the original
distributions for their license terms.

## Contributors

- **Weiwei Jiang** — original Python wrapper
- **SanctusDei** — ML pipeline, hierarchical classifier, StandardScaler, dead sensor detection, task management, battery monitoring, auto-start deployment
