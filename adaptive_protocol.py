"""
adaptive_protocol.py
Closed-Loop Adaptive QKD Protocol Layer.

Runs BB84 sessions with live feature evaluation at dynamic checkpoints, calling the trained
ML threat classifier to adjust protocol behavior in real time:
- CONTINUE: Low risk (P_attack < 0.25) -> Standard privacy amplification
- HARDEN: Medium risk (0.25 <= P_attack < 0.70) -> Increased privacy amplification / aggressive compression
- ABORT: High risk (P_attack >= 0.70) -> Immediate session termination

Includes benchmarking engine comparing Secure Key Rate across No-Defense, Naive Threshold, and Adaptive ML policies.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from bb84_sim import BB84Simulator


def binary_entropy(e: float) -> float:
    """Calculate binary entropy h(e) = -e log2(e) - (1-e) log2(1-e)."""
    if e <= 0.0 or e >= 1.0:
        return 0.0
    return float(-e * np.log2(e) - (1 - e) * np.log2(1 - e))


class AdaptiveQKDProtocol:
    """
    Closed-loop adaptive protocol wrapper for BB84 QKD.
    """
    def __init__(
        self,
        model_path: str = 'model/rf_classifier.pkl',
        preprocessor_path: str = 'model/preprocessor.pkl',
        low_threshold: float = 0.25,
        high_threshold: float = 0.70
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.clf = None
        self.scaler = None
        self.label_encoder = None
        self.feature_cols = None

        self._load_models(model_path, preprocessor_path)

    def _load_models(self, model_path: str, preprocessor_path: str):
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            with open(model_path, 'rb') as f:
                self.clf = pickle.load(f)
            with open(preprocessor_path, 'rb') as f:
                prep = pickle.load(f)
                self.scaler = prep['scaler']
                self.label_encoder = prep['label_encoder']
                self.feature_cols = prep['feature_cols']
        else:
            print("Warning: Model files not found. Using fallback heuristic policy.")

    def classify_checkpoint(self, features: Dict[str, Any]) -> Tuple[str, float, Dict[str, float]]:
        """
        Evaluate checkpoint features using ML classifier.
        Returns:
            predicted_class (str): E.g., 'clean', 'intercept_resend', etc.
            attack_probability (float): Combined probability of threat classes.
            class_probs (dict): Probability breakdown per class.
        """
        if self.clf is None or self.scaler is None:
            # Heuristic fallback if model not loaded yet
            qber = features.get('qber', 0.0)
            p_attack = min(1.0, max(0.0, (qber - 0.04) / 0.20))
            return ('unknown', p_attack, {})

        # Prepare feature vector matching training schema
        vec = np.array([[features[c] for c in self.feature_cols]], dtype=np.float64)
        vec_scaled = self.scaler.transform(vec)

        probs = self.clf.predict_proba(vec_scaled)[0]
        class_probs = {self.label_encoder.classes_[i]: float(probs[i]) for i in range(len(probs))}

        # Threat classes = intercept_resend, pns, noise+attack
        attacked_labels = {'intercept_resend', 'pns', 'noise+attack'}
        p_attack = float(sum(prob for cls_name, prob in class_probs.items() if cls_name in attacked_labels))

        predicted_class = self.label_encoder.classes_[np.argmax(probs)]
        return predicted_class, p_attack, class_probs

    def run_adaptive_session(
        self,
        sim: BB84Simulator,
        noise_prob: float = 0.0,
        attack_type: str = 'clean',
        intercept_prob: float = 0.0,
        pns_split_prob: float = 0.0,
        checkpoints: int = 4
    ) -> Dict[str, Any]:
        """
        Execute a session with multi-stage checkpoints and adaptive risk evaluation.
        """
        session_result = sim.simulate_session_vectorized(
            noise_prob=noise_prob,
            attack_type=attack_type,
            intercept_prob=intercept_prob,
            pns_split_prob=pns_split_prob
        )

        audit_log = []
        final_action = 'CONTINUE'
        final_pa_factor = 1.1

        # Simulate checkpoint feature progression
        raw_qber = session_result['qber']
        sliding_qber = session_result.get('qber_sliding_window', [raw_qber] * 20)

        for cp in range(1, checkpoints + 1):
            fraction = cp / checkpoints
            cp_qber_subset = sliding_qber[:int(len(sliding_qber) * fraction)]
            cp_qber = float(np.mean(cp_qber_subset)) if len(cp_qber_subset) > 0 else raw_qber

            cp_features = session_result.copy()
            cp_features['qber'] = cp_qber
            cp_features['qber_std'] = float(np.std(cp_qber_subset)) if len(cp_qber_subset) > 1 else 0.0

            pred_class, p_attack, class_probs = self.classify_checkpoint(cp_features)

            if p_attack < self.low_threshold:
                action = 'CONTINUE'
                pa_factor = 1.1
            elif p_attack < self.high_threshold:
                action = 'HARDEN'
                pa_factor = 2.2  # Aggressive compression ratio
            else:
                action = 'ABORT'
                pa_factor = 0.0

            audit_log.append({
                'checkpoint': cp,
                'progress_pct': int(fraction * 100),
                'cp_qber': cp_qber,
                'predicted_class': pred_class,
                'p_attack': p_attack,
                'action': action,
                'pa_factor': pa_factor
            })

            final_action = action
            final_pa_factor = pa_factor
            if action == 'ABORT':
                break

        # Compute final secure key rate
        sifted_len = session_result['sifted_key_length']
        qber = session_result['qber']
        h_e = binary_entropy(qber)

        if final_action == 'ABORT':
            final_key_len = 0
            secure_key_rate = 0.0
        elif final_action == 'HARDEN':
            # Hardened privacy amplification compresses key harder
            usable_fraction = max(0.0, 1.0 - final_pa_factor * h_e)
            final_key_len = int(sifted_len * usable_fraction * 0.7)  # Additional discard factor
            secure_key_rate = (final_key_len / session_result['raw_key_length'])
        else:  # CONTINUE
            usable_fraction = max(0.0, 1.0 - final_pa_factor * h_e)
            final_key_len = int(sifted_len * usable_fraction)
            secure_key_rate = (final_key_len / session_result['raw_key_length'])

        session_result['audit_log'] = audit_log
        session_result['final_action'] = final_action
        session_result['final_key_len'] = final_key_len
        session_result['secure_key_rate'] = secure_key_rate

        return session_result


# ==========================================
# Secure Key Rate Benchmark Engine
# ==========================================
def benchmark_defense_policies(
    num_sessions: int = 500,
    model_path: str = 'model/rf_classifier.pkl',
    preprocessor_path: str = 'model/preprocessor.pkl',
    seed: int = 42
) -> pd.DataFrame:
    """
    Headline comparison: Compares Secure Key Rate across 3 defense strategies:
    1. No-Defense (Naive generation, vulnerable to eavesdropping)
    2. Hard-Threshold Defense (Abort whenever QBER > 5%)
    3. Adaptive ML Defense (Uses ML classification)
    """
    protocol = AdaptiveQKDProtocol(model_path=model_path, preprocessor_path=preprocessor_path)
    rng = np.random.default_rng(seed)

    scenarios = [
        ('Clean Channel', 'clean', 0.0, 0.0),
        ('High Channel Noise (8%)', 'noise_only', 0.08, 0.0),
        ('Intercept-Resend (50%)', 'intercept_resend', 0.0, 0.50),
        ('PNS Attack (40%)', 'pns', 0.0, 0.40),
        ('Noise + Intercept (6% noise + 30% attack)', 'noise+attack', 0.06, 0.30)
    ]

    results = []
    for sc_name, attack_type, noise_p, attack_p in scenarios:
        for i in range(num_sessions):
            num_qubits = 1000
            sim = BB84Simulator(num_qubits=num_qubits, seed=seed + i)

            # Adaptive ML Policy
            ad_res = protocol.run_adaptive_session(
                sim=sim, noise_prob=noise_p, attack_type=attack_type,
                intercept_prob=attack_p, pns_split_prob=attack_p
            )
            rate_adaptive = ad_res['secure_key_rate']
            action_adaptive = ad_res['final_action']

            # Base QBER & Sifted length
            qber = ad_res['qber']
            sifted = ad_res['sifted_key_length']
            h_e = binary_entropy(qber)

            # Policy 1: No-Defense (Always try to generate key with standard PA = 1.1)
            # Vulnerable: If attacked, effective key rate is compromised/insecure!
            rate_nodefense = max(0.0, (sifted * (1.0 - 1.1 * h_e)) / num_qubits)
            is_compromised = (attack_type in ('intercept_resend', 'pns', 'noise+attack')) and (rate_nodefense > 0)

            # Policy 2: Hard Threshold Defense (Abort if QBER > 0.05)
            if qber > 0.05:
                rate_hardthresh = 0.0
                action_hardthresh = 'ABORT'
            else:
                rate_hardthresh = max(0.0, (sifted * (1.0 - 1.1 * h_e)) / num_qubits)
                action_hardthresh = 'CONTINUE'

            results.append({
                'scenario': sc_name,
                'attack_type': attack_type,
                'qber': qber,
                'rate_nodefense': rate_nodefense,
                'rate_hardthresh': rate_hardthresh,
                'rate_adaptive': rate_adaptive,
                'action_adaptive': action_adaptive,
                'action_hardthresh': action_hardthresh,
                'is_compromised_nodefense': is_compromised
            })

    df_bench = pd.DataFrame(results)
    return df_bench


if __name__ == '__main__':
    print("Running Adaptive Protocol Benchmark...")
    df_b = benchmark_defense_policies(num_sessions=100)
    summary = df_b.groupby('scenario')[['rate_nodefense', 'rate_hardthresh', 'rate_adaptive']].mean()
    print("\n--- SECURE KEY RATE BENCHMARK SUMMARY (Avg Key Bits / Qubit Sent) ---")
    print(summary)
