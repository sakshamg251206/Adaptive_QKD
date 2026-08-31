"""
generate_dataset.py
Dataset Generator for Adaptive QKD Security.
Simulates 10,000+ BB84 sessions across 5 attack/noise classes with realistic parameter variation,
exporting both tabular feature CSV and time-series sliding-window NPZ files.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from bb84_sim import BB84Simulator

def generate_qkd_dataset(
    num_sessions_per_class: int = 2000,
    output_dir: str = 'data',
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate dataset with stratified representation of all 5 session categories.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    classes = ['clean', 'noise_only', 'intercept_resend', 'pns', 'noise+attack']
    total_sessions = num_sessions_per_class * len(classes)
    print(f"Generating QKD dataset: {total_sessions} sessions total ({num_sessions_per_class} per class)...")

    records: List[Dict[str, Any]] = []
    sequential_qber_list: List[List[float]] = []

    session_id = 0
    for label in classes:
        print(f"  -> Simulating {num_sessions_per_class} sessions for class '{label}'...")
        for i in range(num_sessions_per_class):
            num_qubits = int(rng.integers(800, 1201))
            sim = BB84Simulator(num_qubits=num_qubits, seed=seed + session_id)

            if label == 'clean':
                noise_prob = 0.0
                intercept_prob = 0.0
                pns_split_prob = 0.0
            elif label == 'noise_only':
                # Realistically overlapping depolarizing noise: 1% to 15%
                noise_prob = float(rng.uniform(0.01, 0.15))
                intercept_prob = 0.0
                pns_split_prob = 0.0
            elif label == 'intercept_resend':
                noise_prob = 0.0
                # Interception fraction: 10% to 100%
                intercept_prob = float(rng.uniform(0.10, 1.00))
                pns_split_prob = 0.0
            elif label == 'pns':
                noise_prob = 0.0
                intercept_prob = 0.0
                pns_split_prob = float(rng.uniform(0.15, 0.90))
            elif label == 'noise+attack':
                noise_prob = float(rng.uniform(0.01, 0.12))
                if rng.random() < 0.6:
                    intercept_prob = float(rng.uniform(0.10, 0.80))
                    pns_split_prob = 0.0
                else:
                    intercept_prob = 0.0
                    pns_split_prob = float(rng.uniform(0.15, 0.80))

            res = sim.simulate_session_vectorized(
                noise_prob=noise_prob,
                attack_type=label,
                intercept_prob=intercept_prob,
                pns_split_prob=pns_split_prob,
                num_windows=20,
                test_fraction=0.25
            )

            # Store sliding window separately
            window_qber = res.pop('qber_sliding_window')
            res['session_id'] = session_id

            records.append(res)
            sequential_qber_list.append(window_qber)

            session_id += 1

    df = pd.DataFrame(records)
    seq_matrix = np.array(sequential_qber_list, dtype=np.float32)

    csv_path = os.path.join(output_dir, 'qkd_sessions.csv')
    npz_path = os.path.join(output_dir, 'qkd_sessions_sequential.npz')

    df.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, qber_sequences=seq_matrix, session_ids=df['session_id'].values)

    print(f"\nSuccessfully generated dataset:")
    print(f"  Tabular features saved to:  {csv_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    print(f"  Sequential matrix saved to: {npz_path} ({seq_matrix.shape} shape)")
    print("\nClass distribution:")
    print(df['label'].value_counts())

    return df, seq_matrix

if __name__ == '__main__':
    generate_qkd_dataset(num_sessions_per_class=2000, output_dir='data', seed=42)
