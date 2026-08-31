"""
dashboard/app.py
Streamlit Interactive Dashboard for Adaptive QKD Security via ML-Based Eavesdropping Classification.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bb84_sim import BB84Simulator
from adaptive_protocol import AdaptiveQKDProtocol, benchmark_defense_policies

# Page Configuration
st.set_page_config(
    page_title="Adaptive QKD Security Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #0284C7;
        margin-bottom: 15px;
    }
    .status-continue {
        color: #10B981;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .status-harden {
        color: #F59E0B;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .status-abort {
        color: #EF4444;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_protocol():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'rf_classifier.pkl')
    prep_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'preprocessor.pkl')
    return AdaptiveQKDProtocol(model_path=model_path, preprocessor_path=prep_path)


def main():
    st.title("⚛️ Adaptive QKD Security Dashboard")
    st.markdown("**ML-Based Real-Time Eavesdropping Detection & Dynamic Protocol Defense for BB84 Quantum Key Distribution**")
    st.divider()

    protocol = load_protocol()

    # Sidebar Controls
    st.sidebar.header("🎛️ Session Simulation Controls")
    num_qubits = st.sidebar.slider("Number of Exchanged Qubits", 500, 2000, 1000, step=100)

    attack_type = st.sidebar.selectbox(
        "Ground Truth Attack / Channel Condition",
        ['clean', 'noise_only', 'intercept_resend', 'pns', 'noise+attack']
    )

    noise_prob = st.sidebar.slider("Channel Depolarizing Noise (p_noise)", 0.0, 0.20, 0.04, step=0.01)

    intercept_prob = 0.0
    pns_split_prob = 0.0

    if attack_type in ('intercept_resend', 'noise+attack'):
        intercept_prob = st.sidebar.slider("Eve Interception Fraction (p_intercept)", 0.10, 1.00, 0.50, step=0.05)

    if attack_type in ('pns', 'noise+attack'):
        pns_split_prob = st.sidebar.slider("Eve PNS Split Probability (p_pns)", 0.10, 0.90, 0.40, step=0.05)

    run_sim_btn = st.sidebar.button("🚀 Run Live QKD Session", use_container_width=True)

    # Main Tabs
    tab1, tab2, tab3 = st.tabs([
        "📡 Live Session Monitor",
        "📊 ML Model & Interpretability",
        "🏆 Secure Key Rate Benchmark"
    ])

    # ----------------------------------------------------
    # TAB 1: Live Session Monitor
    # ----------------------------------------------------
    with tab1:
        st.subheader("Live BB84 Quantum Key Distribution Execution")

        sim = BB84Simulator(num_qubits=num_qubits, seed=int(time.time()) % 10000)

        # Run session
        res = protocol.run_adaptive_session(
            sim=sim,
            noise_prob=noise_prob,
            attack_type=attack_type,
            intercept_prob=intercept_prob,
            pns_split_prob=pns_split_prob,
            checkpoints=5
        )

        # Display Top Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Raw Qubits Sent", res['raw_key_length'])
        m2.metric("Sifted Key Bits", res['sifted_key_length'])
        m3.metric("Measured QBER", f"{res['qber'] * 100:.2f}%")
        m4.metric("Timing Jitter", f"{res['timing_jitter_ns']:.2f} ns")

        action = res['final_action']
        if action == 'CONTINUE':
            action_html = '<span class="status-continue">🟢 CONTINUE (Standard PA)</span>'
        elif action == 'HARDEN':
            action_html = '<span class="status-harden">🟡 HARDEN (Aggressive PA)</span>'
        else:
            action_html = '<span class="status-abort">🔴 ABORT (Session Terminated)</span>'

        m5.markdown(f"**Adaptive Action:**<br>{action_html}", unsafe_allow_html=True)

        st.markdown("---")

        c1, c2 = st.columns([7, 5])

        with c1:
            st.markdown("### 📈 Real-Time QBER Sliding Window Stream $Q(t)$")
            sliding_qber = res.get('qber_sliding_window', [res['qber']] * 20)
            df_qber = pd.DataFrame({
                'Time Window': list(range(1, len(sliding_qber) + 1)),
                'Window QBER (%)': [q * 100 for q in sliding_qber]
            })
            st.line_chart(df_qber, x='Time Window', y='Window QBER (%)', height=320)

        with c2:
            st.markdown("### 🔍 Threat Classifier Output")
            # Predict on full session features
            pred_class, p_attack, class_probs = protocol.classify_checkpoint(res)

            st.progress(p_attack, text=f"**Threat Probability (P_attack): {p_attack * 100:.1f}%**")

            st.write(f"**Predicted Threat Class:** `{pred_class}`")
            st.write(f"**Ground-Truth Label:** `{res['label']}`")

            if class_probs:
                df_probs = pd.DataFrame({
                    'Class': list(class_probs.keys()),
                    'Probability': [p * 100 for p in class_probs.values()]
                })
                st.bar_chart(df_probs, x='Class', y='Probability', height=220)

        # Checkpoint Audit Log
        st.markdown("### 📋 Multi-Stage Checkpoint Audit Trail")
        if res.get('audit_log'):
            audit_df = pd.DataFrame(res['audit_log'])
            audit_df['cp_qber'] = audit_df['cp_qber'].apply(lambda x: f"{x*100:.2f}%")
            audit_df['p_attack'] = audit_df['p_attack'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(audit_df, use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: ML Model & Interpretability
    # ----------------------------------------------------
    with tab2:
        st.subheader("Machine Learning Threat Classification Architecture")

        st.markdown("""
        The threat classifier processes statistical features ($QBER$, $QBER_{std}$, $QBER_{max}$, $Sifting Ratio$, $Timing Jitter$)
        to distinguish between natural quantum channel depolarizing noise and targeted eavesdropping attacks.
        """)

        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Confusion Matrix (Random Forest)")
            cm_img = os.path.join(results_dir, 'plots', 'confusion_matrix_rf.png')
            if os.path.exists(cm_img):
                st.image(cm_img, use_container_width=True)

        with col_b:
            st.markdown("#### Binary ROC Curve (Attacked vs Unattacked)")
            roc_img = os.path.join(results_dir, 'plots', 'roc_curve.png')
            if os.path.exists(roc_img):
                st.image(roc_img, use_container_width=True)

        st.divider()

        st.markdown("#### Feature Importance Analysis")
        fi_img = os.path.join(results_dir, 'plots', 'feature_importance.png')
        if os.path.exists(fi_img):
            st.image(fi_img, use_container_width=True)

        # Metrics Markdown Table
        metrics_md_path = os.path.join(results_dir, 'metrics.md')
        if os.path.exists(metrics_md_path):
            with open(metrics_md_path, 'r') as f:
                report_text = f.read()
            st.markdown("### Model Evaluation Metrics Summary")
            st.markdown(report_text)

    # ----------------------------------------------------
    # TAB 3: Secure Key Rate Benchmark
    # ----------------------------------------------------
    with tab3:
        st.subheader("Secure Key Rate Comparison: Naive vs Adaptive vs No-Defense")
        st.markdown("""
        This headline comparison demonstrates that an **Adaptive ML defense policy** optimizes key generation:
        - **No-Defense**: Vulnerable to eavesdropping, continuing session under attack and risking key leakage.
        - **Naive Threshold Defense (Abort if QBER > 5%)**: Overly conservative, discarding valid keys under natural channel noise.
        - **Adaptive ML Policy**: Dynamically hardens or aborts based on threat classification, preserving high secure key rate under noise while protecting confidentiality under attack.
        """)

        if st.button("🔄 Run Full Benchmark Suite", use_container_width=True):
            with st.spinner("Simulating benchmark scenarios across 500 QKD sessions..."):
                df_b = benchmark_defense_policies(num_sessions=100)
                st.session_state['df_bench'] = df_b

        if 'df_bench' in st.session_state:
            df_b = st.session_state['df_bench']
            summary_df = df_b.groupby('scenario')[['rate_nodefense', 'rate_hardthresh', 'rate_adaptive']].mean().reset_index()

            st.markdown("### Benchmark Results (Avg Secure Key Bits / Qubit Sent)")
            st.dataframe(summary_df, use_container_width=True)

            # Bar plot comparison
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(summary_df['scenario']))
            width = 0.25

            ax.bar(x - width, summary_df['rate_nodefense'], width, label='No-Defense (Vulnerable)', color='#94A3B8')
            ax.bar(x, summary_df['rate_hardthresh'], width, label='Naive Threshold (Abort > 5%)', color='#EF4444')
            ax.bar(x + width, summary_df['rate_adaptive'], width, label='Adaptive ML Policy', color='#10B981')

            ax.set_ylabel('Secure Key Rate (bits / qubit)')
            ax.set_title('Secure Key Rate Yield Across Scenarios', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df['scenario'], rotation=25, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()

            st.pyplot(fig)


if __name__ == '__main__':
    main()
