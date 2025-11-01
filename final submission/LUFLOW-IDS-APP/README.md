# LUFLOW Intrusion Detection Desktop Application

A PyQt5-based Windows desktop application that exposes live capture, CSV batch inference, and manual single prediction modes backed by the provided XGBoost model artefacts.

## Features

- **Live Capture Mode** – Uses PyShark/TShark (or synthetic traffic if unavailable) to sniff packets, aggregate them into flows, compute required features, and perform real-time inference with dashboard metrics.
- **CSV Batch Mode** – Load a CSV file aligned to the 15 model features for bulk predictions with alert counts.
- **Single Prediction Mode** – Enter feature values manually for one-off predictions with probability breakdowns.
- **Session Logging** – Each run creates a timestamped folder in `logs/` containing the raw log, exported predictions, and session summary.
- **Exports** – Export session predictions to CSV at any time via the File menu.
- **Developer Tools** – Launch the training notebook from the menu, review logs, and open an About dialog.

## Prerequisites

- Windows 10/11 with Python 3.10+ (tested on conda 3.13 environment).
- Wireshark/TShark (required for live capture; ensure `tshark.exe` is on `PATH`). The application attempts a silent installation via winget on startup if it is missing.
- Npcap (installed by Wireshark) for packet capture on Windows.

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

> **Note**: Live capture requires administrative privileges to access network interfaces.

## Running from Source

```powershell
python main.py
```

## Building a Standalone Executable

Generate a self-contained `.exe` via PyInstaller:

```powershell
pyinstaller --noconfirm --clean ^
    --onefile --windowed ^
    --name "LUFLOW-IDS" ^
    --icon icon.ico ^
    --hidden-import PyQt5.sip ^
    --collect-submodules PyQt5 ^
    --add-data "xgboost_models;xgboost_models" ^
    main.py
```

The built executable resides in `dist/LUFLOW-IDS.exe`. Distribute the entire folder contents to end users.

## Application Structure

```
app/
  config.py              # Paths, constants, metadata
  gui.py                 # PyQt5 main window, tabs, menu actions
  live_capture.py        # Scapy-based capture worker with synthetic fallback
  feature_engineering.py # Flow aggregation + feature extraction helpers
  model_manager.py       # Loads model artefacts and performs inference
  csv_mode.py            # CSV ingestion utilities
  single_prediction.py   # Manual input helper
  session_manager.py     # Session logging + exports
  logging_utils.py       # File/console logger setup
  converters.py          # IP/protocol conversions
xgboost_models/          # Supplied model artefacts (unchanged)
logs/                    # Created on first run with per-session folders
```

## Graceful Degradation

- If Scapy is unavailable, the live capture tab enters a synthetic traffic mode for demonstration purposes.
- Errors are surfaced via GUI dialogs and session logs.
- Manual prediction and CSV modes remain operational regardless of live capture availability.

## Troubleshooting

- **Missing Dependencies** – Use the `pip install -r requirements.txt` command inside your virtual environment. Install Wireshark/TShark so PyShark can access `tshark.exe` for live capture. The app will attempt a one-time silent installation via `winget` when it launches, but this requires administrator rights and the Windows package manager to be available.
- **Permission Errors** – Run the application as Administrator to access network interfaces.
- **Empty Predictions** – Ensure your CSV includes the required feature columns. Missing columns are auto-filled with zero but may reduce accuracy.


