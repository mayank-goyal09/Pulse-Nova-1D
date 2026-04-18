"""
Pulse Nova 1D — ECG Arrhythmia Detection Dashboard
Built from scratch with clinical ECG monitor aesthetics.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import tensorflow as tf
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import ECGPreprocessor

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="Pulse Nova 1D", page_icon="🫀", layout="wide")

# ─────────────────────────────────────────
# MINIMAL BUT EFFECTIVE CUSTOM CSS
# Only inline-style overrides, no class-based CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #13092a; }
    section[data-testid="stSidebar"] { background-color: #1c0f38; }
    .stMetric label { font-size: 0.85rem !important; color: #c4b5fd !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #ede9fe !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    div[data-testid="stExpander"] { border: 1px solid #5b21b6; border-radius: 12px; }
    h1, h2, h3 { color: #ede9fe !important; }
    p, li, span, .stCaption { color: #c4b5fd !important; }
    .stDivider { border-color: #3b1f7a !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# ECG MONITOR PLOT STYLE (Green on Black)
# ─────────────────────────────────────────
ECG_STYLE = {
    'bg': '#1a0d30',
    'grid': '#2e1a54',
    'signal': '#00ff87',
    'signal2': '#22d3ee',
    'peak': '#f472b6',
    'text': '#ede9fe',
    'dim': '#a78bfa',
}

def ecg_monitor_plot(signal, peaks=None, title="ECG Lead II", figsize=(14, 4)):
    """Create an ECG-monitor-style plot (green trace on dark background)."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=ECG_STYLE['bg'])
    ax.set_facecolor(ECG_STYLE['bg'])
    
    time_sec = np.arange(len(signal)) / 360
    ax.plot(time_sec, signal, color=ECG_STYLE['signal'], linewidth=0.9, alpha=0.95)
    
    if peaks is not None and len(peaks) > 0:
        valid = peaks[peaks < len(signal)]
        ax.scatter(valid / 360, signal[valid], color=ECG_STYLE['peak'],
                   s=25, zorder=5, marker='v', label=f'R-peaks ({len(valid)})')
        ax.legend(loc='upper right', fontsize=8, facecolor=ECG_STYLE['bg'],
                  edgecolor='#3b1f7a', labelcolor=ECG_STYLE['text'])
    
    ax.set_title(title, fontsize=13, fontweight='bold', color=ECG_STYLE['text'], pad=12)
    ax.set_xlabel('Time (seconds)', fontsize=10, color=ECG_STYLE['dim'])
    ax.set_ylabel('Amplitude (mV)', fontsize=10, color=ECG_STYLE['dim'])
    ax.tick_params(colors=ECG_STYLE['dim'], labelsize=8)
    
    # ECG grid
    ax.grid(True, which='major', color=ECG_STYLE['grid'], linewidth=0.5, alpha=0.7)
    ax.grid(True, which='minor', color=ECG_STYLE['grid'], linewidth=0.2, alpha=0.3)
    ax.minorticks_on()
    
    for spine in ax.spines.values():
        spine.set_color('#3b1f7a')
    
    plt.tight_layout()
    return fig


def beat_gallery_plot(beats, preds, confs, class_info, count=8):
    """Plot a grid of individual beats, color-coded by class."""
    n = min(count, len(beats))
    cols = 4
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows), facecolor=ECG_STYLE['bg'])
    axes = np.atleast_2d(axes)
    
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.set_facecolor(ECG_STYLE['bg'])
        
        if i < n:
            cls = preds[i]
            color = class_info[cls]['color']
            ax.plot(beats[i], color=color, linewidth=2)
            ax.set_title(f"{class_info[cls]['name']}  ({confs[i]:.0%})",
                         fontsize=9, fontweight='bold', color=color, pad=8)
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#3b1f7a')
    
    plt.tight_layout()
    return fig


def class_distribution_plot(preds, class_info):
    """Horizontal bar chart of beat type counts."""
    counts = np.bincount(preds, minlength=5)
    names = [class_info[i]['name'] for i in range(5)]
    colors = [class_info[i]['color'] for i in range(5)]
    
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=ECG_STYLE['bg'])
    ax.set_facecolor(ECG_STYLE['bg'])
    
    bars = ax.barh(names, counts, color=colors, edgecolor='none', height=0.6)
    
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_width() + max(counts) * 0.02, bar.get_y() + bar.get_height() / 2,
                    str(count), va='center', fontsize=11, fontweight='bold', color=ECG_STYLE['text'])
    
    ax.set_xlabel('Number of Beats', fontsize=10, color=ECG_STYLE['dim'])
    ax.set_title('Beat Classification Distribution', fontsize=13, fontweight='bold',
                 color=ECG_STYLE['text'], pad=12)
    ax.tick_params(colors=ECG_STYLE['dim'], labelsize=10)
    ax.invert_yaxis()
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', color=ECG_STYLE['grid'], linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    return fig


def rr_variability_plot(peaks):
    """Plot R-R intervals to show heart rate variability."""
    if len(peaks) < 3:
        return None
    
    rr = np.diff(peaks) / 360 * 1000  # ms
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), facecolor=ECG_STYLE['bg'])
    
    # Timeline
    ax1.set_facecolor(ECG_STYLE['bg'])
    ax1.plot(rr, color=ECG_STYLE['signal2'], linewidth=1.5, marker='o', markersize=3)
    ax1.axhline(np.mean(rr), color=ECG_STYLE['peak'], linestyle='--', linewidth=1, label=f'Mean: {np.mean(rr):.0f} ms')
    ax1.set_title('R-R Interval Timeline', fontsize=12, fontweight='bold', color=ECG_STYLE['text'])
    ax1.set_xlabel('Beat Number', fontsize=9, color=ECG_STYLE['dim'])
    ax1.set_ylabel('Interval (ms)', fontsize=9, color=ECG_STYLE['dim'])
    ax1.legend(fontsize=8, facecolor=ECG_STYLE['bg'], edgecolor='#3b1f7a', labelcolor=ECG_STYLE['text'])
    ax1.tick_params(colors=ECG_STYLE['dim'], labelsize=8)
    ax1.grid(True, color=ECG_STYLE['grid'], linewidth=0.3)
    for spine in ax1.spines.values():
        spine.set_color('#3b1f7a')
    
    # Histogram
    ax2.set_facecolor(ECG_STYLE['bg'])
    ax2.hist(rr, bins=20, color=ECG_STYLE['signal2'], edgecolor=ECG_STYLE['bg'], alpha=0.8)
    ax2.set_title('R-R Interval Distribution', fontsize=12, fontweight='bold', color=ECG_STYLE['text'])
    ax2.set_xlabel('Interval (ms)', fontsize=9, color=ECG_STYLE['dim'])
    ax2.set_ylabel('Count', fontsize=9, color=ECG_STYLE['dim'])
    ax2.tick_params(colors=ECG_STYLE['dim'], labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#3b1f7a')
    
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# DATA & MODEL
# ─────────────────────────────────────────
CLASS_INFO = {
    0: {'name': 'Normal',          'color': '#10B981', 'emoji': '✅'},
    1: {'name': 'Supraventricular', 'color': '#F59E0B', 'emoji': '⚠️'},
    2: {'name': 'Ventricular',     'color': '#EF4444', 'emoji': '🚨'},
    3: {'name': 'Fusion',          'color': '#A78BFA', 'emoji': '🔀'},
    4: {'name': 'Paced',           'color': '#60A5FA', 'emoji': '🔋'},
}

SAMPLE_CASES = {
    "Patient 100 — Normal Sinus Rhythm": {
        "file": "100.csv",
        "age": 69, "sex": "Male",
        "note": "Healthy control subject with normal heart rhythm throughout the recording."
    },
    "Patient 106 — Ventricular Ectopy (PVCs)": {
        "file": "106.csv",
        "age": 24, "sex": "Female",
        "note": "Frequent premature ventricular contractions. A good example of abnormal ventricular activity."
    },
    "Patient 119 — Ventricular Bigeminy": {
        "file": "119.csv",
        "age": 51, "sex": "Female",
        "note": "Every other beat is a PVC (bigeminy pattern). Clinically significant arrhythmia."
    },
    "Patient 208 — Complex Arrhythmia": {
        "file": "208.csv",
        "age": 63, "sex": "Female",
        "note": "Multiple types of ventricular ectopy including PVCs. Complex case for analysis."
    },
}

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_ecg_model.h5')
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_preprocessor():
    return ECGPreprocessor(fs=360, window_size=180)


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
def main():
    # ── SIDEBAR ──
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=60)
        st.title("Pulse Nova 1D")
        st.caption("Deep Learning ECG Analyzer")
        st.divider()
        
        source = st.radio("Data Source", ["📤 Upload CSV", "🏥 Case Library (Local Only)"])
        
        # Link to GitHub Repository
        st.markdown("[📂 GitHub Repository](https://github.com/mayank-goyal09/Pulse-Nova-1D.git)")
        
        signal = None
        patient = {}
        
        if source == "🏥 Case Library (Local Only)":
            case_name = st.selectbox("Select Patient", list(SAMPLE_CASES.keys()))
            case = SAMPLE_CASES[case_name]
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
            path = os.path.join(data_dir, case['file'])
            
            if os.path.exists(path):
                df = pd.read_csv(path)
                df.columns = [c.strip("'") for c in df.columns]
                signal = df['MLII'].values
                patient = case
                patient['id'] = case['file'].replace('.csv', '')
            else:
                st.error(f"File not found: {path}")
            
            st.info(f"📝 {case['note']}")
        else:
            uploaded = st.file_uploader("Upload ECG CSV", type=['csv'])
            if uploaded:
                df = pd.read_csv(uploaded)
                df.columns = [c.strip("'") for c in df.columns]
                if 'MLII' in df.columns:
                    signal = df['MLII'].values
                else:
                    signal = df.iloc[:, 1].values
                patient = {'id': 'UPLOAD', 'age': '—', 'sex': '—', 'note': 'User-uploaded recording'}
        
        st.divider()
        st.caption("Built with TensorFlow · MIT-BIH Database")
        st.caption("96.48% Test Accuracy")

    # ── TITLE ──
    st.title("🫀 Pulse Nova 1D — ECG Analysis Report")
    st.caption("Automated arrhythmia detection powered by a 1D Convolutional Neural Network trained on the MIT-BIH Arrhythmia Database.")
    
    if signal is None:
        st.warning("👈 Select a patient case from the sidebar or upload a CSV file to begin.")
        
        # Landing page with educational content
        st.divider()
        st.header("📚 Understanding ECG & Arrhythmias")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("What is an ECG?")
            st.write("""
            An **Electrocardiogram (ECG)** is a test that records the electrical activity of your heart. 
            Each heartbeat is triggered by an electrical impulse that travels through the heart muscle, 
            causing it to contract and pump blood.
            
            A normal ECG has these key waves:
            - **P Wave**: Atrial contraction (upper chambers)
            - **QRS Complex**: Ventricular contraction (lower chambers) — the tall spike
            - **T Wave**: Ventricular recovery
            
            The shape, timing, and regularity of these waves tell doctors if your heart is working normally.
            """)
        
        with col2:
            st.subheader("What is an Arrhythmia?")
            st.write("""
            An **arrhythmia** is an abnormal heart rhythm. It happens when the electrical signals 
            that coordinate heartbeats don't work properly.
            
            **Types we detect:**
            
            | Type | What It Means | Risk |
            |------|---------------|------|
            | Normal (N) | Healthy beat | None ✅ |
            | Supraventricular (S) | Beat from upper chambers | Low ⚠️ |
            | Ventricular (V) | Beat from lower chambers | High 🚨 |
            | Fusion (F) | Mixed origin beat | Medium 🔀 |
            | Paced (Q) | Pacemaker beat | Expected 🔋 |
            """)
        
        st.divider()
        st.subheader("🤖 How Our AI Works")
        st.write("""
        1. **Signal Filtering**: We remove noise using a bandpass filter (0.5–40 Hz)
        2. **Beat Detection**: We find each heartbeat by detecting R-peaks (the tall spikes)
        3. **Segmentation**: Each beat is extracted as a 180-sample window
        4. **Classification**: A 1D-CNN neural network classifies each beat into 5 categories
        5. **Report Generation**: Results are compiled into this clinical dashboard
        """)
        
        st.info("💡 **Fun Fact**: Our AI independently learned to focus on the QRS complex width — the exact same feature cardiologists use to diagnose ventricular arrhythmias!")
        return

    # ── LOAD & ANALYZE ──
    model = load_model()
    preprocessor = load_preprocessor()
    
    with st.spinner("🔄 Analyzing ECG signal..."):
        filtered = preprocessor.apply_filter(signal)
        beats, peaks = preprocessor.segment_beats(filtered)
        
        if len(beats) == 0:
            st.error("Could not detect any heartbeats in this signal.")
            return
        
        X = beats.reshape(-1, 180, 1)
        probs = model.predict(X, verbose=0)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
    
    # ── STATS ──
    counts = np.bincount(preds, minlength=5)
    total = len(preds)
    abnormal = total - counts[0]
    burden = (abnormal / total) * 100
    avg_conf = np.mean(confs) * 100
    
    # Estimate heart rate from R-R intervals
    if len(peaks) > 1:
        rr_sec = np.diff(peaks) / 360
        hr = 60 / np.mean(rr_sec)
    else:
        hr = 0

    st.divider()
    
    # ── SECTION 1: Patient Summary ──
    st.header("📋 Patient Summary")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Patient ID", f"#{patient.get('id', '?')}")
    c2.metric("Age / Sex", f"{patient.get('age', '?')} / {patient.get('sex', '?')[0] if patient.get('sex') else '?'}")
    c3.metric("Avg Heart Rate", f"{hr:.0f} BPM")
    c4.metric("Total Beats", f"{total}")
    c5.metric("Arrhythmia Burden", f"{burden:.1f}%", delta=f"{abnormal} abnormal" if abnormal > 0 else "All normal")

    # ── SECTION 2: AI Verdict ──
    if burden < 1:
        st.success(f"✅ **Normal Sinus Rhythm** — The AI detected {counts[0]}/{total} normal beats. No arrhythmia detected.")
    elif burden < 10:
        st.warning(f"⚠️ **Minor Irregularities** — {abnormal} abnormal beats detected ({burden:.1f}% burden). Occasional ectopy may be benign, but monitoring is recommended.")
    else:
        st.error(f"🚨 **Significant Arrhythmia** — {abnormal} abnormal beats detected ({burden:.1f}% burden). This level of ectopy warrants further clinical evaluation.")

    st.divider()
    
    # ── SECTION 3: ECG Signal ──
    st.header("📈 ECG Signal Viewer")
    st.caption("Green trace = filtered ECG signal. Red markers = detected R-peaks (heartbeats).")
    
    # Show first 10 seconds
    display_len = min(3600, len(filtered))
    display_peaks = peaks[peaks < display_len]
    fig = ecg_monitor_plot(filtered[:display_len], display_peaks,
                           title=f"ECG Lead II — Patient #{patient.get('id', '?')} (First 10 seconds)")
    st.pyplot(fig)
    plt.close()
    
    with st.expander("💡 How to read this chart"):
        st.write("""
        - **Tall sharp spikes** (R-peaks) = Individual heartbeats
        - **Regular spacing** between peaks = Normal rhythm
        - **Irregular spacing** or **unusual shapes** = Possible arrhythmia
        - **Red triangles** = AI-detected R-peaks used for segmentation
        """)

    st.divider()
    
    # ── SECTION 4: Classification Results ──
    st.header("🎯 Classification Results")
    st.caption("Each heartbeat was classified by the neural network into one of 5 AAMI categories.")
    
    # Class counts
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            emoji = CLASS_INFO[i]['emoji']
            name = CLASS_INFO[i]['name']
            count = counts[i]
            pct = (count / total * 100) if total > 0 else 0
            st.metric(f"{emoji} {name}", f"{count}", f"{pct:.1f}%")
    
    # Distribution chart
    fig = class_distribution_plot(preds, CLASS_INFO)
    st.pyplot(fig)
    plt.close()

    st.divider()
    
    # ── SECTION 5: Beat Gallery ──
    st.header("🫀 Individual Beat Morphology")
    st.caption("Sample heartbeat waveforms extracted from the recording, color-coded by classification.")
    
    # Let user pick which class to view
    view_class = st.selectbox("Filter by beat type", 
                               ["All Types"] + [f"{CLASS_INFO[i]['emoji']} {CLASS_INFO[i]['name']}" for i in range(5)])
    
    if view_class == "All Types":
        # Show diverse sample
        sample_idx = np.random.choice(len(beats), min(8, len(beats)), replace=False)
        sample_idx = np.sort(sample_idx)
    else:
        cls_idx = [i for i in range(5) if CLASS_INFO[i]['name'] in view_class][0]
        class_mask = np.where(preds == cls_idx)[0]
        sample_idx = class_mask[:8] if len(class_mask) > 0 else []
    
    if len(sample_idx) > 0:
        fig = beat_gallery_plot(beats[sample_idx], preds[sample_idx], confs[sample_idx], CLASS_INFO)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No beats of this type detected in the recording.")
    
    with st.expander("💡 Understanding beat shapes"):
        st.write("""
        - **Normal beats** have a sharp, narrow QRS spike followed by a smooth T-wave
        - **Ventricular beats (PVCs)** have a wide, bizarre QRS that looks very different from normal
        - **Supraventricular beats** look similar to normal but may have a premature timing
        - **Fusion beats** are a mix — one half looks normal, the other half looks ventricular
        - **Paced beats** have a sharp pacemaker spike before the QRS complex
        """)

    st.divider()
    
    # ── SECTION 6: Heart Rate Variability ──
    st.header("💓 Heart Rate Variability (HRV)")
    st.caption("R-R interval analysis reveals the regularity of the heartbeat.")
    
    fig = rr_variability_plot(peaks)
    if fig:
        st.pyplot(fig)
        plt.close()
        
        rr = np.diff(peaks) / 360 * 1000
        hrv_cols = st.columns(4)
        hrv_cols[0].metric("Mean R-R", f"{np.mean(rr):.0f} ms")
        hrv_cols[1].metric("Std Dev (SDNN)", f"{np.std(rr):.1f} ms")
        hrv_cols[2].metric("Min R-R", f"{np.min(rr):.0f} ms")
        hrv_cols[3].metric("Max R-R", f"{np.max(rr):.0f} ms")
        
        with st.expander("💡 What does HRV mean?"):
            st.write("""
            **Heart Rate Variability (HRV)** measures the variation in time between heartbeats.
            
            - **High HRV** (more variation) → Generally healthy. Your heart adapts well to changes.
            - **Low HRV** (very regular) → May indicate stress, fatigue, or certain conditions.
            - **Irregular spikes** in the R-R timeline → May indicate premature beats (PVCs/PACs).
            
            **SDNN** (Standard Deviation of R-R intervals) is the most common HRV metric:
            - SDNN > 100 ms = Healthy
            - SDNN 50-100 ms = Moderate
            - SDNN < 50 ms = Low (may need evaluation)
            """)

    st.divider()
    
    # ── SECTION 7: Clinical Findings ──
    st.header("📝 Detailed Clinical Report")
    
    report_lines = []
    report_lines.append(f"**Patient**: #{patient.get('id', 'Unknown')} | Age: {patient.get('age', '?')} | Sex: {patient.get('sex', '?')}")
    report_lines.append(f"**Recording**: 360 Hz, Lead MLII")
    report_lines.append(f"**Analysis Date**: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    report_lines.append("")
    report_lines.append(f"**Total Beats Analyzed**: {total}")
    report_lines.append(f"**Average Heart Rate**: {hr:.0f} BPM")
    report_lines.append(f"**Average AI Confidence**: {avg_conf:.1f}%")
    report_lines.append("")
    
    # Beat breakdown
    report_lines.append("### Beat Classification Breakdown")
    for i in range(5):
        if counts[i] > 0:
            pct = counts[i] / total * 100
            report_lines.append(f"- **{CLASS_INFO[i]['name']}**: {counts[i]} beats ({pct:.1f}%)")
    
    report_lines.append("")
    
    # Interpretation
    report_lines.append("### Clinical Interpretation")
    
    if burden < 1:
        report_lines.append("The recording shows **normal sinus rhythm** throughout. No significant ectopy or conduction abnormalities were detected by the AI model.")
        report_lines.append("")
        report_lines.append("**Recommendation**: No immediate action required. Routine follow-up as clinically indicated.")
    else:
        report_lines.append(f"The recording shows an **arrhythmia burden of {burden:.1f}%** ({abnormal} out of {total} beats).")
        report_lines.append("")
        
        if counts[2] > 0:
            pvc_pct = counts[2] / total * 100
            report_lines.append(f"**Ventricular Ectopy**: {counts[2]} premature ventricular contractions (PVCs) detected ({pvc_pct:.1f}%). "
                               f"PVCs originate from the ventricles and bypass normal conduction. "
                               f"{'This burden level (>10%) may be clinically significant and warrants echocardiographic evaluation.' if pvc_pct > 10 else 'Occasional PVCs are common and often benign in structurally normal hearts.'}")
            report_lines.append("")
        
        if counts[1] > 0:
            report_lines.append(f"**Supraventricular Ectopy**: {counts[1]} premature atrial contractions (PACs) detected. "
                               f"These originate above the ventricles (atria or AV node). "
                               f"Frequent PACs may predispose to atrial fibrillation in some patients.")
            report_lines.append("")
        
        if counts[3] > 0:
            report_lines.append(f"**Fusion Beats**: {counts[3]} detected. These occur when a normal impulse and a ventricular impulse collide. "
                               f"Presence of fusion beats often accompanies ventricular ectopy.")
            report_lines.append("")
        
        if counts[4] > 0:
            report_lines.append(f"**Paced Rhythm**: {counts[4]} pacemaker-initiated beats detected. "
                               f"This is expected in patients with implanted pacemakers. "
                               f"Pacemaker function appears {'normal' if counts[4] / total > 0.5 else 'intermittent (sensing/pacing may need review)'}.")
            report_lines.append("")
        
        report_lines.append("**Recommendation**: Clinical correlation is advised. Consider Holter monitoring for longer-term rhythm assessment if not already performed.")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*This report was auto-generated by Pulse Nova 1D (v2.1). "
                       "It is intended for educational/research purposes only and should not replace professional medical diagnosis.*")
    
    full_report = "\n".join(report_lines)
    st.markdown(full_report)
    
    # Download button
    st.download_button(
        label="📥 Download Report as Text",
        data=full_report,
        file_name=f"ecg_report_{patient.get('id', 'unknown')}_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown"
    )

    st.divider()
    
    # ── SECTION 8: Educational Footer ──
    st.header("📚 Learn More About ECG Analysis")
    
    with st.expander("🫀 What is the AAMI Standard?"):
        st.write("""
        The **Association for the Advancement of Medical Instrumentation (AAMI)** 
        defines a standard way to group ECG beat types for automated analysis:
        
        | AAMI Class | Original MIT-BIH Labels | Description |
        |------------|------------------------|-------------|
        | N (Normal) | N, L, R, e, j | Any beat from the sinus node or normal conduction |
        | S (Supraventricular) | A, a, J, S | Premature beats from above the ventricles |
        | V (Ventricular) | V, E | Premature beats from the ventricles |
        | F (Fusion) | F | Combination of normal and ventricular |
        | Q (Paced) | / , f, p | Pacemaker-generated beats |
        
        This grouping is the gold standard for ECG classification research.
        """)
    
    with st.expander("🧠 About Our 1D-CNN Model"):
        st.write("""
        **Architecture**: 3-layer 1D Convolutional Neural Network
        
        ```
        Input (180 samples, 1 channel)
            → Conv1D(32, kernel=5) → BatchNorm → MaxPool
            → Conv1D(64, kernel=5) → BatchNorm → MaxPool → Dropout(0.3)
            → Conv1D(128, kernel=3) → BatchNorm → GlobalAvgPool
            → Dense(64) → Dropout(0.4) → Dense(5, Softmax)
        ```
        
        **Training Details**:
        - Dataset: MIT-BIH Arrhythmia Database (48 patients, ~80,000+ beats)
        - Class Weights: Balanced (to handle medical data imbalance)
        - Early Stopping: Patience=7 with best-weight restoration
        - Test Accuracy: **96.48%**
        
        **Key Insight**: The model learned to focus on QRS complex width — the same 
        diagnostic feature cardiologists use — without being explicitly told to do so.
        """)
    
    with st.expander("⚕️ Medical Disclaimer"):
        st.write("""
        This application is developed for **educational and research purposes only**.
        
        - It has NOT been validated for clinical diagnostic use.
        - It should NOT replace professional medical judgment.
        - Always consult a qualified healthcare provider for medical advice.
        - The MIT-BIH database represents a limited patient population and may not generalize to all populations.
        
        If you experience chest pain, shortness of breath, or irregular heartbeats, 
        please seek immediate medical attention.
        """)


if __name__ == "__main__":
    main()
