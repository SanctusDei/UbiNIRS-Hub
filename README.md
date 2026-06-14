# UbiNIRS-Hub

**A self-contained handheld NIR sensing platform that runs the full
acquisition → preprocessing → inference → field-adaptation pipeline entirely
on-device — no cloud, no tethering, no lab hardware.**

Built on Raspberry Pi 3A+ with a TI DLP NIRScan Nano and 5-inch touchscreen.
Published at ACM UbiComp/ISWC 2026.
<br><sub>Jiahao Gong · Xurui Li · Weiwei Jiang</sub>

---

## Hardware

| Component | Details |
|-----------|---------|
| Spectrometer | TI DLP NIRScan Nano (900–1700 nm, USB HID) |
| Compute | Raspberry Pi 3A+ (BCM2837B0, ARM Cortex-A53) |
| Display | 5-inch DSI touchscreen (800×480) |
| Battery Monitor | INA219 over I²C (addr 0x42) — optional |
| Power | 2S LiPo (7.4 V nominal, ~1800 mAh) — optional |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/SanctusDei/UbiNIRS-Hub.git
cd UbiNIRS-Hub

# 2. Install dependencies
sudo apt-get install libudev-dev libusb-1.0-0-dev
pip install numpy scipy pandas scikit-learn joblib

# 3. Compile the native library (Raspberry Pi)
chmod +x rebuild_so.sh && ./rebuild_so.sh

# 4. Launch
sudo python3 main.py
```

To auto-start on boot: `chmod +x setup_autostart.sh && ./setup_autostart.sh`

> **Note**: The NIRScan Nano requires **root** for USB HID access. See
> [pyusb/libusb permissions](https://stackoverflow.com/questions/3738173).

---

## Features

### Spectral Acquisition
- Hadamard & Column scan modes, configurable 900–1700 nm range
- PGA gain control, lamp on/off, hibernation
- Real-time absorbance / reflectance / intensity graph
- SNR diagnostics and dead-sensor detection
- Auto-save to timestamped CSVs (`data/spectra/`)

### On-Device ML
- **Classifiers**: SVM, Random Forest, KNN, LDA
- **Regressors**: SVR, PLS
- **Preprocessing pipeline**: SNV → Savitzky-Golay → derivative → scaling
  (configurable per algorithm)
- **Hierarchical classification**: two-level HIGH/LOW split via raw absorbance
  mean, with per-branch classifiers and confidence scores
- **Dual-gate rejection**: classifier confidence threshold + PCA reconstruction-error
  gate — rejects out-of-distribution samples as `UNKNOWN`
- **Stratified memory**: bounded replay buffer (≤100 samples, ≥3 per class)
  with proportional pruning — prevents catastrophic forgetting
- **On-device field adaptation**: incremental retraining without cloud connectivity

### Task Management
- SQLite-backed CRUD with card-based touch browser
- Classification & regression task types with tag-based organization
- LOOCV evaluation, batch prediction with majority voting
- Model export/import (joblib `.pkl`)

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                SpectrometerApp                   │
│    Main window · hardware heartbeat thread       │
├─────────────────────────────────────────────────┤
│  ScanWorkflowMixin     (gui_scan.py)             │
│  MLEngineMixin         (gui_ml.py)               │
│  TaskManagerMixin      (gui_tasks.py)            │
├─────────────────────────────────────────────────┤
│  NIRS.py  ←  _NIRScanner.so  ←  C++ SWIG layer  │
│  INA219.py                                   │
│  SQLite (spectral_tasks.db)                  │
└─────────────────────────────────────────────────┘
```

### Key Design Decisions

| Paper § | Concept | Implementation |
|---------|---------|---------------|
| §2.1 | Hardware heartbeat | Background poller thread; auto-disables UI on disconnect, restores on reconnect |
| §2.2 | Preprocessing defaults | SNV + Savitzky-Golay 1st-derivative (window=11, polyorder=3) |
| §2.3.1 | Stratified memory | ≤100 samples; per-class min³; proportional pruning on overflow |
| §2.3.2 | Hierarchical classifier | HIGH/LOW splits on absorbance mean; per-branch SVC/KNN/RF |
| §2.3.2 | Dual-gate rejection | `confidence_score > θ` ∧ `PCA_recon_error < τ` → accept; else → UNKNOWN |
| §2.2 | Dead-sensor detection | Rejects scans with peak-to-peak intensity < 1e-6 |

---

## Project Structure

```
UbiNIRS-Hub/
├── main.py               Entry point
├── gui_app.py            Main GUI (SpectrometerApp)
├── gui_scan.py           Scan workflow mixin
├── gui_ml.py             ML engine mixin
├── gui_tasks.py          Task management mixin
├── NIRS.py               Python wrapper for _NIRScanner.so
├── INA219.py             INA219 battery driver
├── rebuild_so.sh         Recompile native library
├── setup_autostart.sh    Boot-time auto-launch
│
├── src/                  C++ source + SWIG interface
│   ├── NIRScanner.i      SWIG definition
│   ├── NIRScanner.cpp/h  Python–C++ bridge
│   ├── API.cpp/h         High-level scan API
│   ├── dlpspec_*.c/h     TI DLP Spectrum Library
│   ├── hid.c/hidapi.h    USB HID transport
│   └── scripts/          Build scripts (py2/py3)
│
├── models/               Trained model pickles (.pkl)
└── data/
    └── spectra/          Timestamped spectral CSVs
```

---

## Citation

```bibtex
@inproceedings{gong2026ubinirshub,
  author    = {Gong, Jiahao and Li, Xurui and Jiang, Weiwei},
  title     = {UbiNIRS-Hub: A Mobile Near-Infrared Sensing Platform for
               On-Device Field Adaptation in Ubiquitous Material Analysis},
  booktitle = {Proceedings of the ACM International Joint Conference on
               Pervasive and Ubiquitous Computing (UbiComp/ISWC '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {Shanghai, China},
  doi       = {10.1145/XXXXXXX.XXXXXXX}
}
```

### Related Work

```bibtex
@article{jiang2020probing,
  author  = {Jiang, Weiwei and Marini, Gabriele and van Berkel, Niels
             and Sarsenbayeva, Zhanna and Tan, Zheyu and Luo, Chu and
             He, Xin and Dingler, Tilman and Goncalves, Jorge and
             Kawahara, Yoshihiro and Kostakos, Vassilis},
  title   = {Probing Sucrose Contents in Everyday Drinks Using
             Miniaturized Near-Infrared Spectroscopy Scanners},
  journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  volume  = {3},
  number  = {4},
  year    = {2020},
  doi     = {10.1145/3369834}
}

@article{jiang2022near,
  author  = {Jiang, Weiwei and Yu, Difeng and Wang, Chaofan and
             Sarsenbayeva, Zhanna and van Berkel, Niels and
             Goncalves, Jorge and Kostakos, Vassilis},
  title   = {Near-infrared Imaging for Information Embedding and
             Extraction with Layered Structures},
  journal = {ACM Trans. Graph.},
  volume  = {42},
  number  = {1},
  year    = {2022},
  doi     = {10.1145/3533426}
}
```

---

## Related Repositories

- [NIRScanner-Imaging](https://github.com/HighTemplar-wjiang/NIRScanner-Imaging) — no-reference wavelength selection for NIR imaging
- [NIRScanner-Plotter](https://github.com/HighTemplar-wjiang/NIRScanner-Plotter) — Django server controlling NIRScan Nano + 2D plotter

---

## License

Includes TI DLP NIRscan Nano GUI and DLP Spectrum Library source. Refer to the
original distributions for their license terms.

## Contributors

- **Weiwei Jiang** — original Python wrapper
- **SanctusDei** — ML pipeline, hierarchical classifier, dual-gate rejection, task management, battery monitoring, auto-start deployment
