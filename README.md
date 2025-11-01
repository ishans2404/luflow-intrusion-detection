<div align="center">

![LUFLOW-IDS Logo](final%20submission/LUFLOW-IDS-APP/icon.ico)

# 🛡️ LUFLOW Network Intrusion Detection System

**Production-Grade Flow-Based IDS with 95% Accuracy and Sub-Millisecond Inference**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()

*Edge-optimized intrusion detection achieving Random Forest accuracy (94.97%) with XGBoost speed (0.028ms latency)*

[🚀 Download Windows App](#-windows-application) • [📊 View Results](#-performance-results) • [📖 Read Documentation](#-documentation) • [🎓 Academic Report](final%20submission/project-report/project-report-22BCE2608.pdf)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Performance Results](#-performance-results)
- [Project Architecture](#-project-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Model Training](#-model-training)
- [Windows Application](#-windows-application)
- [Documentation](#-documentation)
- [Repository Structure](#-repository-structure)
- [Citation](#-citation)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

LUFLOW-IDS is a **production-ready network intrusion detection system** developed through a rigorous three-phase research methodology:

- **Phase I:** Dataset engineering assembling 7.89M flows from LUFlow repository
- **Phase II:** Multi-model benchmarking comparing Random Forest, XGBoost, and LightGBM
- **Phase III:** Hyperparameter optimization and Windows desktop application deployment

The system achieves **95.40% classification accuracy** with **0.028ms inference latency**, making it suitable for **edge deployment** on resource-constrained hardware (Raspberry Pi 4, IoT gateways).

### 🎓 Academic Context

This project represents the **comprehensive thesis submission** for **B.Tech Computer Science with Specialization in Bioinformatics** at **Vellore Institute of Technology**, demonstrating end-to-end machine learning engineering from dataset assembly to production deployment.

---

## ✨ Key Features

### 🔬 Research Contributions

- ✅ **Production-Scale Dataset Engineering:** 7.89M network flows with balanced temporal coverage (24 months)
- ✅ **Rigorous Multi-Model Benchmarking:** Systematic comparison of Random Forest, XGBoost, LightGBM
- ✅ **Edge-Optimized Performance:** <200MB memory footprint, sub-millisecond inference
- ✅ **Hyperparameter Optimization:** 4.27pp accuracy improvement through RandomizedSearchCV
- ✅ **Complete Deployment Pipeline:** PyQt5 GUI, PyInstaller packaging, GitHub releases

### 🖥️ Application Capabilities

- 🔴 **Live Capture Mode:** Real-time packet capture via TShark/PyShark integration
- 📁 **CSV Batch Processing:** Bulk flow file analysis with automated feature alignment
- 🎯 **Single Flow Prediction:** Manual feature input for individual flow classification
- 📊 **Session Management:** Automatic logging and CSV export of predictions
- 🚀 **Multi-Mode Operation:** Graceful degradation when dependencies unavailable

---

## 📊 Performance Results

### Model Comparison Summary

| Model | Accuracy | F1-Score | Latency (ms) | Memory (MB) | Throughput (samples/s) |
|-------|----------|----------|--------------|-------------|------------------------|
| **Random Forest** | **94.97%** | **0.9512** | 0.0114 | 318.76 | 87,719 |
| **XGBoost (Optimized)** | **95.40%** | **0.9531** | **0.0280** | **195.64** | **35,714** |
| XGBoost (Baseline) | 91.13% | 0.9048 | **0.0030** | 195.64 | **332,666** |
| LightGBM | 90.91% | 0.9132 | 0.0137 | 391.24 | 72,992 |

### Per-Class Performance (Optimized XGBoost)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Benign** | 1.00 | 1.00 | 1.00 | 848,656 |
| **Malicious** | 0.91 | 0.95 | 0.93 | 505,862 |
| **Outlier** | 0.86 | 0.76 | 0.81 | 199,346 |

### Key Achievements

- ✅ **95.40% overall accuracy** (exceeding 90% threshold by 5.4pp)
- ✅ **0.028ms inference latency** (178× faster than 5ms requirement)
- ✅ **196MB peak memory** (2.6× below 500MB constraint)
- ✅ **Perfect benign classification** (100% precision/recall)
- ✅ **58% outlier recall improvement** through optimization

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PHASE I: DATA ENGINEERING                   │
├─────────────────────────────────────────────────────────────┤
│  LUFlow Repository (241 files) → Balanced Selection         │
│  → 7.89M flows assembled → Quality Assurance (1.54% missing)│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PHASE II: MULTI-MODEL BENCHMARKING              │
├─────────────────────────────────────────────────────────────┤
│  Train/Test Split (80/20 stratified) → Model Training       │
│  → Performance Evaluation → Feature Importance Analysis     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│        PHASE III: OPTIMIZATION & DEPLOYMENT                  │
├─────────────────────────────────────────────────────────────┤
│  RandomizedSearchCV (50 iterations) → XGBoost Optimization  │
│  → PyQt5 GUI → PyInstaller Packaging → GitHub Release      │
└─────────────────────────────────────────────────────────────┘
```

### 🛠️ Technology Stack

**Machine Learning:**
- XGBoost 2.0.3 (Gradient Boosting)
- scikit-learn 1.5.2 (Random Forest, preprocessing)
- LightGBM 4.3.0 (Histogram-based boosting)

**Data Processing:**
- pandas 2.2.3 (DataFrame operations)
- NumPy 1.26.4 (Numerical computing)

**Application Development:**
- PyQt5 5.15.10 (GUI framework)
- PyShark 0.6 (Packet capture integration)
- PyInstaller 5.13 (Executable packaging)

**Deployment:**
- Joblib 1.4.2 (Model serialization)
- Windows 10/11 (Target platform)

---

## 🚀 Installation

### Option 1: Windows Executable (Recommended)

**Download the pre-built Windows application:**

1. Download [LUFLOW-IDS.exe](https://github.com/ishans2404/luflow-intrusion-detection/blob/7cfde34688251ec735002e4a145a03ab7e397179/final%20submission/LUFLOW-IDS-APP/dist/LUFLOW-IDS.exe)
2. Run the executable (no installation required)

**Requirements:**
- Windows 10/11 (64-bit)
- 4GB RAM (minimum), 8GB recommended
- Administrator privileges (for live capture mode)
- TShark/Wireshark (optional, for live capture)

### Option 2: Python Source Installation

**Clone the repository:**

```
git clone https://github.com/yourusername/luflow-intrusion-detection.git
cd luflow-intrusion-detection
```

**Create virtual environment:**

```
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

**Install dependencies:**

```
cd "final submission/LUFLOW-IDS-APP"
pip install -r requirements.txt
```

**Run the application:**

```
python main.py
```

---

## 📖 Usage

### Windows Application

**Launch the application:**

```
# From release
.\LUFLOW-IDS.exe

# From source
python main.py
```

### Three Operational Modes

#### 1. 🔴 Live Capture Mode

**Real-time network flow monitoring:**

- Captures packets via TShark/PyShark
- Aggregates flows with configurable timeout
- Performs real-time classification
- Displays predictions with probabilities

**Requirements:**
- TShark installed and available on PATH
- Administrator/root privileges
- Active network interface

**Graceful Degradation:**
- Falls back to synthetic traffic generation if TShark unavailable
- Demonstrates functionality without live capture dependencies

#### 2. 📁 CSV Batch Processing

**Bulk flow file analysis:**

```
1. Click "CSV Batch Mode" tab
2. Select CSV file containing network flows
3. Application automatically aligns features
4. View predictions in table with confidence scores
5. Export results to timestamped CSV file
```

**CSV Format Requirements:**
- Columns matching Joy feature schema (15 features)
- Missing columns filled with default values
- Automatic feature name mapping

**Example CSV:**

```
src_ip,src_port,dest_ip,dest_port,proto,bytes_in,bytes_out,num_pkts_in,num_pkts_out,entropy,total_entropy,avg_ipt,time_start,time_end,duration
192.168.1.100,54321,8.8.8.8,53,17,150,450,3,3,2.5,3.1,0.01,1609459200.0,1609459201.5,1.5
10.0.0.50,443,172.217.0.1,80,6,5000,120000,50,100,7.8,7.9,0.02,1609459300.0,1609459350.0,50.0
```

#### 3. 🎯 Single Flow Prediction

**Manual feature input for individual classification:**

- Enter 15 network flow features via GUI form
- Real-time probability display (benign/malicious/outlier)
- Educational demonstration of model behavior
- Useful for sanity checking and debugging

### Command-Line Notebook Execution

**Run Jupyter notebooks for model training:**

```
# Phase I: Dataset preparation
jupyter notebook network-intrusion-dataset-preparation.ipynb

# Phase II: Multi-model benchmarking
jupyter notebook network-intrusion-modelling.ipynb

# Phase III: XGBoost optimization
jupyter notebook network-intrusion-xgboost.ipynb
```

---

## 📦 Dataset

### LUFlow Repository

The project utilizes the **LUFlow network flow dataset** from Lancaster University:

- **Source:** [LUFlow Network Intrusion Detection Data Set](https://www.kaggle.com/datasets/mryanm/luflow-network-intrusion-detection-data-set)
- **Total Files:** 241 CSV files discovered
- **Selected Files:** 135 files (balanced temporal selection)
- **Total Flows:** 7,890,694 assembled
- **Temporal Coverage:** June 2020 - June 2022 (24 months)
- **Classes:** Benign (53.8%), Malicious (33.3%), Outlier (12.9%)

### Feature Schema (15 Joy Features)

| Feature | Type | Description |
|---------|------|-------------|
| `src_ip` | String | Source IP address |
| `src_port` | Integer | Source port number |
| `dest_ip` | String | Destination IP address |
| `dest_port` | Integer | Destination port number |
| `proto` | Integer | Protocol (TCP=6, UDP=17) |
| `bytes_in` | Integer | Inbound bytes |
| `bytes_out` | Integer | Outbound bytes |
| `num_pkts_in` | Integer | Inbound packet count |
| `num_pkts_out` | Integer | Outbound packet count |
| `entropy` | Float | Payload entropy |
| `total_entropy` | Float | Total flow entropy |
| `avg_ipt` | Float | Average inter-packet time |
| `time_start` | Float | Flow start timestamp |
| `time_end` | Float | Flow end timestamp |
| `duration` | Float | Flow duration (seconds) |

### Sample Data

A sample dataset is provided in `sample_flows.csv` for testing:

```
head sample_flows.csv
```

---

## 🎓 Model Training

### Phase I: Dataset Preparation

**Objective:** Assemble production-scale dataset from distributed LUFlow repository

```
# Key operations performed in notebook
- File discovery: 241 CSV files identified
- Balanced selection: 135 files chosen (max 11.3% per month)
- Quality assurance: 121,376 missing values removed (1.54%)
- Final dataset: 7,769,318 clean flows
```

**Run the notebook:**

```
jupyter notebook network-intrusion-dataset-preparation.ipynb
```

### Phase II: Multi-Model Benchmarking

**Objective:** Compare Random Forest, XGBoost, LightGBM under standardized conditions

```
# Evaluated models with results
1. Random Forest: 94.97% accuracy (accuracy leader)
2. XGBoost: 91.13% accuracy (speed champion, 0.003ms)
3. LightGBM: 90.91% accuracy (balanced performance)
```

**Run the notebook:**

```
jupyter notebook network-intrusion-modelling.ipynb
```

### Phase III: XGBoost Optimization

**Objective:** Hyperparameter tuning and deployment artifact generation

```
# RandomizedSearchCV configuration
- Parameter combinations explored: 50
- Cross-validation folds: 3 (StratifiedKFold)
- Optimization metric: Weighted F1-score
- Search time: 353.46 seconds
- Best CV F1: 0.9107
- Final test accuracy: 95.40%
```

**Run the notebook:**

```
jupyter notebook network-intrusion-xgboost.ipynb
```

**Generated artifacts:**

```
xgboost_models/
├── optimized_xgboost_luflow.pkl    # Trained model (8.77 MB)
├── label_encoder.pkl                # Class label encoder
├── feature_names.pkl                # Feature column ordering
├── model_metadata.pkl               # Training metadata
└── inference_pipeline.py            # Standalone inference helper
```

---

## 🖥️ Windows Application

### Building the Executable

**Build using PyInstaller:**

```
cd "final submission/LUFLOW-IDS-APP"
.\build_exe.ps1
```

**Manual build command:**

```
pyinstaller --noconfirm --clean --onefile --windowed \
  --name LUFLOW-IDS \
  --icon icon.ico \
  --hidden-import PyQt5.sip \
  --collect-submodules PyQt5 \
  --add-data "xgboost_models;xgboost_models" \
  main.py
```

**Output:**
- `dist/LUFLOW-IDS.exe` (standalone executable)
- `build/` (PyInstaller intermediate files)

### Application Architecture

```
LUFLOW-IDS-APP/
├── main.py                          # Application entry point
├── app/
│   ├── gui.py                       # PyQt5 main window
│   ├── model_manager.py             # Model loading & inference
│   ├── live_capture.py              # PyShark integration
│   ├── csv_mode.py                  # Batch CSV processing
│   ├── single_prediction.py         # Manual input mode
│   └── session_manager.py           # Logging & export
├── xgboost_models/                  # Model artifacts
└── requirements.txt                 # Python dependencies
```

### Configuration

Edit `app/config.py` to customize:

```
# Model paths
MODEL_DIR = "xgboost_models"

# Live capture settings
CAPTURE_TIMEOUT = 30  # seconds
PACKET_COUNT = 100

# Logging
LOG_DIR = "logs"
EXPORT_FORMAT = "csv"
```

---

## 📚 Documentation

### Academic Report

**Comprehensive thesis document:** [project-report-22BCE2608.pdf](final%20submission/project-report/project-report-22BCE2608.pdf)

**Contents:**
- Chapter 1: Introduction and motivation
- Chapter 2: Literature survey
- Chapter 3: Methodology
- Chapter 4: Phase I - Dataset preparation
- Chapter 5: Phase II - Model benchmarking
- Chapter 6: Phase III - Optimization & deployment
- Chapter 7: Experiment design
- Chapter 8: Results and analysis
- Chapter 9: Conclusions and future work


### API Documentation

**Inference Pipeline Usage:**

```
from xgboost_models.inference_pipeline import create_inference_pipeline

# Load pipeline
pipeline = create_inference_pipeline("xgboost_models")

# Predict on DataFrame
results = pipeline(flow_dataframe)

# Access predictions
predictions = results['predictions']  # Class labels
probabilities = results['probabilities']  # Confidence scores
labels = results['labels']  # Decoded class names
```

---

## 📁 Repository Structure

```
luflow-intrusion-detection/
├── README.md                        # This file
├── LICENSE                          # MIT License
├── .gitignore                       # Git ignore patterns
│
├── final submission/
│   ├── LUFLOW-IDS-APP/              # 🖥️ Windows application
│   │   ├── app/                     # Application modules
│   │   ├── xgboost_models/          # Model artifacts
│   │   ├── dist/LUFLOW-IDS.exe      # Compiled executable
│   │   ├── main.py                  # Entry point
│   │   └── requirements.txt         # Dependencies
│   │
│   └── project-report/              # 📖 Academic thesis
│       ├── chapters/                # LaTeX chapters (10 files)
│       ├── images/                  # Figures and plots
│       ├── main.tex                 # Main LaTeX document
│       ├── references.bib           # Bibliography
│       └── project-report.pdf       # Compiled thesis
│
├── network-intrusion-dataset-preparation.ipynb  # Phase I notebook
├── network-intrusion-modelling.ipynb            # Phase II notebook
├── network-intrusion-xgboost.ipynb              # Phase III notebook
│
├── phase-1-submission/              # Phase 1 deliverables
├── phase-2-submission/              # Phase 2 deliverables
│   ├── assets/                      # Visualization images
│   └── Phase_II_Report.md           # Detailed report
│
└── sample_flows.csv                 # Sample dataset
```

---

## 📊 Benchmarking

### Hardware Configuration

**Development Environment:**
- **Platform:** Kaggle Notebook
- **CPU:** Intel Xeon (2× vCPU)
- **RAM:** 13GB available
- **Storage:** 73GB temporary
- **Python:** 3.11.13

**Target Deployment:**
- **Device:** Raspberry Pi 4 (4GB RAM)
- **OS:** Windows 10/11 or Linux
- **Network:** Gigabit Ethernet

### Performance Profiling

**Training Time:**
- Random Forest: 818.87s (13.6 min)
- XGBoost: 145.27s (2.4 min) ⚡
- LightGBM: 156.86s (2.6 min)
- XGBoost Hyperparameter Search: 353.46s (5.9 min)

**Inference Throughput:**
- Best: XGBoost Baseline (332,666 samples/s)
- Optimized: XGBoost Optimized (35,714 samples/s)
- Balanced: Random Forest (87,719 samples/s)

---

## 🤝 Contributing

Contributions welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create feature branch:** `git checkout -b feature/amazing-feature`
3. **Commit changes:** `git commit -m 'Add amazing feature'`
4. **Push to branch:** `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Setup

```
# Clone your fork
git clone https://github.com/yourusername/luflow-intrusion-detection.git
cd luflow-intrusion-detection

# Create development environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install jupyter pytest black flake8

# Run tests
pytest tests/

# Format code
black app/
flake8 app/
```

---

## 📝 Citation

If you use this work in your research, please cite:

```
@misc{singh2025luflow,
  author       = {Ishan Singh},
  title        = {Flow-Based Network Intrusion Detection: 
                  Edge-Optimized Machine Learning Approach},
  year         = {2025},
  howpublished = {Undergraduate Project Report, Vellore Institute of Technology},
  note         = {Computer Science and Engineering}
}
```

### Related Publications

- **LUFlow Dataset:** Lancaster University Network Flow Repository
- **XGBoost:** Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
- **Random Forest:** Breiman, L. (2001). Random forests. *Machine learning*, 45(1), 5-32.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


---

## 🙏 Acknowledgments

- **LUFlow Dataset:** Lancaster University Intrusion Detection Research Group
- **Vellore Institute of Technology:** Academic supervision and infrastructure
- **Kaggle:** Free computational resources for model training
- **Open Source Community:** XGBoost, scikit-learn, PyQt5 developers

---

## 🔮 Future Work

### Planned Enhancements

- 🔄 **Multi-Platform Support:** Linux and macOS executable builds
- 🧠 **Deep Learning Models:** LSTM/GRU for temporal sequence modeling
- 🛡️ **Adversarial Robustness:** Evasion attack testing and hardening
- 🌐 **REST API:** Flask/FastAPI backend for distributed deployment
- 📱 **Mobile App:** Android/iOS alert notification companion
- 📊 **Dashboard:** Real-time monitoring with Grafana integration

### Research Directions

- Federated learning for privacy-preserving cross-organization training
- Explainability integration (SHAP, LIME) for prediction interpretation
- Zero-day detection using unsupervised anomaly detection
- Encrypted traffic classification (TLS 1.3, QUIC)

---

<div align="center">

**⭐ Star this repository if you find it useful! ⭐**

**Questions? Open an [Issue](https://github.com/ishans2404/luflow-intrusion-detection/issues)**


</div>
