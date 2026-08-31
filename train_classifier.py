"""
train_classifier.py
ML Eavesdropping Classifier for Adaptive QKD Security.

Trains and evaluates:
1. Random Forest Classifier on static session feature vectors.
2. PyTorch 1D-CNN / LSTM Sequence Classifier on sliding-window QBER time-series.

Outputs model artifacts to model/ and comprehensive metric reports + plots to results/.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure reproducible random state
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================
# 1. PyTorch LSTM Sequence Classifier Model
# ==========================================
class QBERSequenceLSTM(nn.Module):
    """
    LSTM sequence classifier that processes sliding-window QBER sequences Q(t)
    to detect temporal attack patterns (e.g., localized bursts or PNS signatures).
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_classes: int = 5, num_layers: int = 2):
        super(QBERSequenceLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use final hidden state from top layer
        last_hidden = h_n[-1]
        out = self.dropout(self.relu(self.fc1(last_hidden)))
        logits = self.fc2(out)
        return logits


# ==========================================
# 2. Main Training & Evaluation Pipeline
# ==========================================
def train_and_evaluate_models(data_dir: str = 'data', model_dir: str = 'model', results_dir: str = 'results'):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    csv_path = os.path.join(data_dir, 'qkd_sessions.csv')
    npz_path = os.path.join(data_dir, 'qkd_sessions_sequential.npz')

    if not os.path.exists(csv_path) or not os.path.exists(npz_path):
        raise FileNotFoundError("Dataset files not found. Please run generate_dataset.py first.")

    # Load dataset
    df = pd.read_csv(csv_path)
    npz_data = np.load(npz_path)
    seq_matrix = npz_data['qber_sequences']  # Shape: (N, 20)

    print(f"Loaded tabular dataset: {df.shape[0]} rows.")
    print(f"Loaded sequential matrix: {seq_matrix.shape} shape.")

    # Define feature set and target label
    feature_cols = [
        'qber', 'qber_std', 'qber_min', 'qber_max',
        'sifted_key_length', 'sifting_ratio',
        'basis_mismatch_rate', 'timing_jitter_ns'
    ]

    X_tab = df[feature_cols].values
    y_raw = df['label'].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)
    print(f"Classes ({len(class_names)}): {class_names}")

    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
        X_tab, y, seq_matrix, test_size=0.20, random_state=SEED, stratify=y
    )

    # Standardize Tabular Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save Preprocessing Artifacts
    with open(os.path.join(model_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump({'scaler': scaler, 'label_encoder': label_encoder, 'feature_cols': feature_cols}, f)

    # ----------------------------------------------------
    # MODEL 1: Random Forest Baseline
    # ----------------------------------------------------
    print("\n==========================================")
    print("Training Model 1: Random Forest Baseline")
    print("==========================================")
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=SEED, n_jobs=-1)
    rf_clf.fit(X_train_scaled, y_train)

    y_pred_rf = rf_clf.predict(X_test_scaled)
    y_prob_rf = rf_clf.predict_proba(X_test_scaled)

    rf_acc = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Test Accuracy: {rf_acc * 100:.2f}%")

    with open(os.path.join(model_dir, 'rf_classifier.pkl'), 'wb') as f:
        pickle.dump(rf_clf, f)

    # ----------------------------------------------------
    # MODEL 2: PyTorch LSTM Sequence Classifier
    # ----------------------------------------------------
    print("\n==========================================")
    print("Training Model 2: PyTorch LSTM Sequence Model")
    print("==========================================")

    # Reshape sequence data: (N, seq_len, 1)
    seq_train_t = torch.tensor(seq_train, dtype=torch.float32).unsqueeze(-1)
    seq_test_t = torch.tensor(seq_test, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(seq_train_t, y_train_t)
    test_dataset = TensorDataset(seq_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using PyTorch device: {device}")

    lstm_model = QBERSequenceLSTM(input_dim=1, hidden_dim=64, num_classes=len(class_names), num_layers=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(lstm_model.parameters(), lr=0.003, weight_decay=1e-4)

    epochs = 25
    for epoch in range(epochs):
        lstm_model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = lstm_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f}")

    # Evaluate LSTM
    lstm_model.eval()
    lstm_preds = []
    lstm_probs = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            logits = lstm_model(batch_x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            lstm_preds.extend(preds.cpu().numpy())
            lstm_probs.extend(probs.cpu().numpy())

    y_pred_lstm = np.array(lstm_preds)
    y_prob_lstm = np.array(lstm_probs)

    lstm_acc = accuracy_score(y_test, y_pred_lstm)
    print(f"PyTorch LSTM Test Accuracy: {lstm_acc * 100:.2f}%")

    torch.save(lstm_model.state_dict(), os.path.join(model_dir, 'lstm_classifier.pt'))

    # ----------------------------------------------------
    # 3. Evaluation & Comparative Analysis
    # ----------------------------------------------------
    print("\n==========================================")
    print("Generating Classification Metrics & Visuals")
    print("==========================================")

    # Confusion Matrix (Random Forest)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Random Forest Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', 'confusion_matrix_rf.png'), dpi=300)
    plt.close()

    # Binary ROC-AUC (Attacked vs Clean/Noise)
    # Define binary target: 1 = Attacked (intercept_resend, pns, noise+attack), 0 = Unattacked (clean, noise_only)
    attacked_labels = {'intercept_resend', 'pns', 'noise+attack'}
    y_test_binary = np.array([1 if class_names[i] in attacked_labels else 0 for i in y_test])

    prob_attack_rf = np.sum(y_prob_rf[:, [class_names.index(c) for c in attacked_labels]], axis=1)
    prob_attack_lstm = np.sum(y_prob_lstm[:, [class_names.index(c) for c in attacked_labels]], axis=1)

    auc_rf = roc_auc_score(y_test_binary, prob_attack_rf)
    auc_lstm = roc_auc_score(y_test_binary, prob_attack_lstm)

    fpr_rf, tpr_rf, _ = roc_curve(y_test_binary, prob_attack_rf)
    fpr_lstm, tpr_lstm, _ = roc_curve(y_test_binary, prob_attack_lstm)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest Baseline (AUC = {auc_rf:.4f})', linewidth=2)
    plt.plot(fpr_lstm, tpr_lstm, label=f'PyTorch LSTM Model (AUC = {auc_lstm:.4f})', linewidth=2, linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('ROC Curve: Binary Threat Detection (Attacked vs Unattacked)', fontsize=13, fontweight='bold')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', 'roc_curve.png'), dpi=300)
    plt.close()

    # Feature Importance Plot (Random Forest Permutation / Gini Importance)
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(9, 5))
    plt.bar([feature_cols[i] for i in indices], importances[indices], color='teal', alpha=0.85)
    plt.title('Random Forest Feature Importance Analysis', fontsize=13, fontweight='bold')
    plt.ylabel('Gini Feature Importance')
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', 'feature_importance.png'), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # 4. Save Markdown Summary Report
    # ----------------------------------------------------
    report_md_path = os.path.join(results_dir, 'metrics.md')
    rf_rep = classification_report(y_test, y_pred_rf, target_names=class_names, output_dict=True)
    lstm_rep = classification_report(y_test, y_pred_lstm, target_names=class_names, output_dict=True)

    with open(report_md_path, 'w') as f:
        f.write("# QKD Eavesdropping Classifier Evaluation Report\n\n")
        f.write("## Overview & Model Performance\n")
        f.write(f"- **Random Forest Accuracy**: `{rf_acc * 100:.2f}%` | **Binary ROC-AUC**: `{auc_rf:.4f}`\n")
        f.write(f"- **PyTorch LSTM Accuracy**: `{lstm_acc * 100:.2f}%` | **Binary ROC-AUC**: `{auc_lstm:.4f}`\n\n")

        f.write("## Random Forest Classification Metrics per Class\n\n")
        f.write("| Class Name | Precision | Recall | F1-Score | Support |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for name in class_names:
            metrics = rf_rep[name]
            f.write(f"| `{name}` | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {int(metrics['support'])} |\n")

        f.write("\n## PyTorch LSTM Classification Metrics per Class\n\n")
        f.write("| Class Name | Precision | Recall | F1-Score | Support |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for name in class_names:
            metrics = lstm_rep[name]
            f.write(f"| `{name}` | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {int(metrics['support'])} |\n")

        f.write("\n## Feature Importance Analysis\n\n")
        f.write("Ranked feature contribution from Gini Importance:\n\n")
        for rank, idx in enumerate(indices, 1):
            f.write(f"{rank}. **{feature_cols[idx]}**: `{importances[idx]:.4f}`\n")

        f.write("\n\n*Report automatically generated by `train_classifier.py`*\n")

    print(f"\nTraining & Evaluation complete! Summary report saved to {report_md_path}")
    print(f"Plots saved to {os.path.join(results_dir, 'plots')}")


if __name__ == '__main__':
    train_and_evaluate_models()
