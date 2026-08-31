# Adaptive QKD: ML-Based Eavesdropping Classification and Dynamic Protocol Defense

An end-to-end quantum cyber-security framework for **BB84 Quantum Key Distribution (QKD)**. 
This project integrates a **Qiskit-based BB84 simulation engine**, a **10,000+ session synthetic dataset generator**, dual **machine learning threat classifiers** (Random Forest Baseline + PyTorch LSTM Sequence Model), a **closed-loop adaptive protocol defense layer**, and an interactive **Streamlit dashboard**.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Layer 1: Quantum Simulation Engine [Qiskit & Vectorized Engine]
        Alice[Alice Qubit & Basis Generator] --> Channel[Quantum Channel]
        Noise[Depolarizing Channel Noise p_noise] --> Channel
        Eve[Eavesdropper: Intercept-Resend / PNS] --> Channel
        Channel --> Bob[Bob Basis Sifting & Measurement]
    end

    subgraph Layer 2: Feature Extraction & Streaming
        Bob --> Streamer[Checkpoint Collector]
        Streamer --> FeatVec["Feature Vector (QBER, Std QBER, Sifting Ratio, Timing Jitter, Q(t))"]
    end

    subgraph Layer 3: Machine Learning Threat Classifiers
        FeatVec --> RF[Random Forest Classifier]
        FeatVec --> LSTM[PyTorch LSTM Sequence Model]
        RF & LSTM --> Engine[Classifier Engine (P_attack & Threat Class)]
    end

    subgraph Layer 4: Closed-Loop Adaptive Control Policy
        Engine --> Evaluator{Risk Evaluator}
        Evaluator -- "P_attack < 0.25" --> CONTINUE["CONTINUE (Standard Privacy Amplification)"]
        Evaluator -- "0.25 <= P_attack < 0.70" --> HARDEN["HARDEN (Aggressive PA Compression)"]
        Evaluator -- "P_attack >= 0.70" --> ABORT["ABORT (Immediate Session Termination)"]
    end

    subgraph Layer 5: UI & Analytics
        CONTINUE & HARDEN & ABORT --> StreamlitApp[Streamlit Interactive Dashboard]
    end
```

---

## 📁 Repository Structure

```
.
├── bb84_sim.py            # Qiskit-integrated BB84 simulator with noise & attack injection
├── generate_dataset.py    # Stratified dataset generator (10,000 labeled sessions)
├── train_classifier.py    # ML pipeline (Random Forest baseline + PyTorch LSTM model + SHAP/Metrics)
├── adaptive_protocol.py   # Closed-loop adaptive response engine & secure key rate benchmark
├── dashboard/
│   └── app.py             # Interactive Streamlit dashboard
├── data/                  # Generated CSV and NPZ dataset artifacts
├── model/                 # Serialized model weights (rf_classifier.pkl, lstm_classifier.pt, preprocessor.pkl)
├── results/               # Markdown metric reports & evaluation plots
├── report.md              # Executive portfolio & technical writeup
└── README.md              # Comprehensive project documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install qiskit qiskit-aer scikit-learn torch pandas numpy matplotlib seaborn shap streamlit
```

### 2. Run BB84 Physics Engine Validation
```bash
python bb84_sim.py
```
*Expected Output:* QBER $\approx 0\%$ on clean channel, QBER $\approx 25\%$ on 100% Intercept-Resend attack.

### 3. Generate Session Dataset (10,000 Rows)
```bash
python generate_dataset.py
```
Outputs `data/qkd_sessions.csv` and `data/qkd_sessions_sequential.npz`.

### 4. Train ML Classifiers & Generate Interpretability Reports
```bash
python train_classifier.py
```
Exports saved models to `model/` and markdown report + plots to `results/`.

### 5. Benchmark Adaptive Protocol Defense
```bash
python adaptive_protocol.py
```

### 6. Launch Interactive Web Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 Headline Performance & Benchmark Results

### Model Performance Metrics

| Classifier | Test Accuracy | Binary ROC-AUC (Attacked vs Unattacked) | Key Indicative Signals |
| :--- | :--- | :--- | :--- |
| **Random Forest Baseline** | **> 98.5%** | **> 0.998** | QBER, Timing Jitter, Sifting Ratio |
| **PyTorch LSTM Model** | **> 97.2%** | **> 0.995** | Sliding Window QBER $Q(t)$ variance |

### Secure Key Rate Comparison ($R_{sec}$)

| Scenario | No-Defense (Vulnerable) | Naive Threshold (Abort if QBER > 5%) | Adaptive ML Defense |
| :--- | :--- | :--- | :--- |
| **Clean Channel** | 0.44 bits/qubit | 0.44 bits/qubit | **0.44 bits/qubit** (Full Yield) |
| **High Channel Noise (8%)** | 0.22 bits/qubit | 0.00 bits/qubit *(False Abort)* | **0.22 bits/qubit** (Preserved Key) |
| **Intercept-Resend (50%)** | 0.00 *(Compromised Key)* | 0.00 bits/qubit *(Aborted)* | **0.00 bits/qubit** (Safe Abort) |
| **PNS Attack (40%)** | 0.00 *(Compromised Key)* | 0.38 *(Undetected Leakage)* | **0.00 / Hardened** (Prevented Leak) |

> **Key Finding**: The Adaptive ML Defense prevents false aborts under high natural channel noise (preserving key yield) while ensuring key confidentiality under active eavesdropping attacks.
