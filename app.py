# -*- coding: utf-8 -*-
"""
app.py — Streamlit Deployment: Audio Speech Classification
============================================================
Mendukung 5 varian model:
  1. RF   + MFCC
  2. CNN  + MelSpec
  3. CNN  + MFCC
  4. LSTM + MelSpec
  5. LSTM + MFCC

Input  : upload file audio (wav/mp3/flac/ogg/m4a) atau rekaman langsung
         (mutual exclusive — hanya satu sumber aktif dalam satu sesi)
Output : prediksi kelas + confidence score + visualisasi pipeline interaktif
"""

import io
import json
import os
import re
import tempfile
import time
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import noisereduce as nr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from pydub import AudioSegment
matplotlib.use("Agg")


# ===========================================================================
# [DEV] KONFIGURASI PATH MODEL
# ===========================================================================

MODEL_DIR = Path("./models")


# ===========================================================================
# [DEV] KONSTANTA FITUR
# ===========================================================================
# Nilai default. Pada saat aplikasi dijalankan, nilai-nilai ini ditimpa
# oleh model_metadata.json (di-load oleh load_all_models) agar identik
# dengan konfigurasi yang digunakan pada tahap pelatihan. Default di sini
# berfungsi sebagai fallback dan memungkinkan modul ini di-import tanpa
# error sebelum metadata tersedia.

AUDIO_SR       = 16000
AUDIO_DURATION = 5.0
N_MFCC         = 40
N_MELS         = 128
HOP_LENGTH     = 512
N_FFT          = 2048
FMAX           = 8000
AUDIO_EXTS     = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm")


def _apply_metadata_to_globals(metadata: dict) -> None:
    """
    Sinkronkan konstanta pipeline dengan model_metadata.json.

    Dipanggil sekali pada saat load_all_models() selesai memuat metadata.
    Memastikan AUDIO_SR, AUDIO_DURATION, N_MFCC, N_MELS, HOP_LENGTH,
    N_FFT, dan FMAX ekuivalen dengan AudioCfg dan FeatureCfg pada tahap
    pelatihan, sehingga pipeline preprocessing dan ekstraksi fitur pada
    inferensi identik dengan pelatihan.
    """
    global AUDIO_SR, AUDIO_DURATION
    global N_MFCC, N_MELS, HOP_LENGTH, N_FFT, FMAX

    audio_cfg   = metadata.get("audio_cfg",   {})
    feature_cfg = metadata.get("feature_cfg", {})

    AUDIO_SR       = int(audio_cfg.get("sr",            AUDIO_SR))
    AUDIO_DURATION = float(audio_cfg.get("duration_s",  AUDIO_DURATION))
    N_MFCC         = int(feature_cfg.get("n_mfcc",      N_MFCC))
    N_MELS         = int(feature_cfg.get("n_mels",      N_MELS))
    HOP_LENGTH     = int(feature_cfg.get("hop_length",  HOP_LENGTH))
    N_FFT          = int(feature_cfg.get("n_fft",       N_FFT))
    FMAX           = int(feature_cfg.get("fmax",        FMAX))


# ===========================================================================
# [DEV] REGISTRI VARIAN MODEL
# ===========================================================================

VARIANT_OPTIONS = {
    "RF + MFCC":      {"model_key": "rf_mfcc",     "arch": "rf",   "feature": "mfcc"},
    "CNN + MelSpec":  {"model_key": "cnn_melspec",  "arch": "cnn",  "feature": "melspec"},
    "CNN + MFCC":     {"model_key": "cnn_mfcc",     "arch": "cnn",  "feature": "mfcc"},
    "LSTM + MelSpec": {"model_key": "lstm_melspec", "arch": "lstm", "feature": "melspec"},
    "LSTM + MFCC":    {"model_key": "lstm_mfcc",    "arch": "lstm", "feature": "mfcc"},
}


# ===========================================================================
# [DEV] PALET WARNA TEMA
# ===========================================================================

BG_MAIN  = "#0b0f1a"
BG_CARD  = "#111827"
CLR_CYAN = "#00e5cc"
CLR_VIOL = "#a78bfa"
CLR_AMBR = "#f97316"
CLR_MUTE = "#64748b"
CLR_TEXT = "#e2e8f0"

CMAP_SPEC = LinearSegmentedColormap.from_list(
    "Javan Sparrow Audio Classifier",
    [BG_MAIN, "#1a2a3a", CLR_CYAN, CLR_VIOL, CLR_AMBR],
)


# ===========================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ===========================================================================

st.set_page_config(
    page_title="Javan Sparrow Audio Classifier",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# [DEV] CSS KUSTOM
# ===========================================================================

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,600;1,300&display=swap');

  :root {
    --bg-main:    #0b0f1a;
    --bg-card:    #111827;
    --accent-1:   #00e5cc;
    --accent-2:   #f97316;
    --accent-3:   #a78bfa;
    --text-main:  #e2e8f0;
    --text-muted: #64748b;
    --border:     rgba(0,229,204,0.15);
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-main);
    color: var(--text-main);
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #0b1320 100%);
    border-right: 1px solid var(--border);
  }

  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00e5cc 0%, #a78bfa 60%, #f97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 0.2rem;
  }

  .hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* Tombol sumber aktif / non-aktif */
  .src-btn-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
  }

  .src-btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.55rem 0;
    border-radius: 8px;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.18s;
  }

  .src-btn-active {
    background: rgba(0,229,204,0.12);
    border-color: var(--accent-1);
    color: var(--accent-1);
  }

  .src-btn-inactive {
    background: rgba(255,255,255,0.03);
    border-color: rgba(255,255,255,0.08);
    color: var(--text-muted);
  }

  /* Badge pill sumber aktif di atas pratinjau */
  .source-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  }

  .source-badge-upload {
    background: rgba(249,115,22,0.12);
    border: 1px solid rgba(249,115,22,0.4);
    color: #f97316;
  }

  .source-badge-record {
    background: rgba(0,229,204,0.10);
    border: 1px solid rgba(0,229,204,0.35);
    color: #00e5cc;
  }

  /* Banner "ganti sumber" */
  .swap-hint {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }

  .swap-link {
    color: var(--accent-1);
    text-decoration: underline;
    cursor: pointer;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
  }

  /* Step cards */
  .step-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
  }

  .step-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent-1), var(--accent-3));
    border-radius: 3px 0 0 3px;
  }

  .step-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent-1);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
  }

  .step-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-main);
  }

  .model-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    border: 1px solid var(--accent-1);
    color: var(--accent-1);
    letter-spacing: 0.1em;
    background: rgba(0,229,204,0.05);
  }

  .pred-box {
    background: linear-gradient(135deg,
      rgba(0,229,204,0.08) 0%,
      rgba(167,139,250,0.08) 100%);
    border: 1px solid rgba(0,229,204,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
  }

  .pred-class {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-1);
    letter-spacing: 0.05em;
  }

  .pred-conf {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
  }

  div.stButton > button {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: linear-gradient(135deg, #00e5cc, #a78bfa);
    color: #0b0f1a;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1.8rem;
    font-weight: 700;
    white-space: nowrap;        /* teks tidak di-wrap ke baris baru */
    transition: all 0.2s;
  }

  div.stButton > button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,229,204,0.3);
  }

  /* Tombol swap (ganti sumber) — lebar mengikuti konten */
  .swap-hint div.stButton > button {
    padding: 0.25rem 0.7rem;
    font-size: 0.7rem;
    background: transparent;
    color: #00e5cc;
    border: 1px solid rgba(0,229,204,0.35);
    white-space: nowrap;
  }

  /* Disabled state — jelas terlihat tidak aktif */
  div.stButton > button:disabled {
    background: #1e2d3d !important;
    color: #334155 !important;
    border: 1px solid #1e2d3d !important;
    box-shadow: none !important;
    transform: none !important;
    cursor: not-allowed !important;
    opacity: 0.6;
  }

  [data-testid="stFileUploader"] {
    background: transparent;
    border-radius: 10px;
    border: 1px dashed var(--border);
  }

  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# PEMUATAN MODEL
# ===========================================================================

_KERAS_UNKNOWN_LAYER_KEYS = frozenset({
    "quantization_config",
    "lora_rank",
    "lora_enabled",
})


def _strip_keras_compat(obj: object) -> object:
    """
    Hapus secara rekursif kunci konfigurasi Keras yang tidak dikenal oleh
    versi lama dari seluruh struktur config (dict / list bersarang).
    """
    if isinstance(obj, dict):
        return {
            k: _strip_keras_compat(v)
            for k, v in obj.items()
            if k not in _KERAS_UNKNOWN_LAYER_KEYS
        }
    if isinstance(obj, list):
        return [_strip_keras_compat(item) for item in obj]
    return obj


def _load_keras_model_compat(path: str):
    """
    Muat model Keras dengan penanganan kompatibilitas lintas-versi.

    Strategi (berurutan — berhenti di langkah pertama yang berhasil):
      1. tf.keras.models.load_model(compile=False) — path normal.
      2. Patch config.json di dalam arsip .keras, lalu muat ulang.
    """
    import zipfile
    import tensorflow as tf

    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e_normal:
        first_error = str(e_normal)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            patched_path = os.path.join(tmp_dir, "model_patched.keras")
            with zipfile.ZipFile(path, "r") as zin:
                with zipfile.ZipFile(patched_path, "w", zipfile.ZIP_DEFLATED) as zout:
                    for item in zin.infolist():
                        raw = zin.read(item.filename)
                        if item.filename == "config.json":
                            cfg = json.loads(raw.decode("utf-8"))
                            cfg = _strip_keras_compat(cfg)
                            raw = json.dumps(cfg, ensure_ascii=False).encode("utf-8")
                        zout.writestr(item, raw)
            return tf.keras.models.load_model(patched_path, compile=False)
    except Exception as e_patch:
        raise RuntimeError(
            f"Gagal memuat model dari '{path}'.\n"
            f"  • Normal load : {first_error}\n"
            f"  • Patched load: {e_patch}\n\n"
            "Pastikan versi TensorFlow/Keras di environment ini kompatibel "
            "dengan versi yang digunakan saat training."
        ) from e_patch


@st.cache_resource(show_spinner=False)
def load_all_models(model_dir: Path):
    """
    Muat semua artefak model dari MODEL_DIR.
    Dipanggil sekali per sesi Streamlit berkat @st.cache_resource.
    """
    errors          = []
    loaded          = {}
    meta_path       = model_dir / "model_metadata.json"
    label_meta_path = model_dir.parent / "meta" / "label_meta.json"

    if not meta_path.exists():
        st.error(
            f"❌ `model_metadata.json` tidak ditemukan di `{model_dir}`.\n\n"
            "Pastikan konstanta `MODEL_DIR` di `app.py` sudah benar."
        )
        return None

    with open(meta_path) as f:
        metadata = json.load(f)

    # Sinkronkan konstanta global dengan konfigurasi pelatihan sebelum
    # pipeline preprocessing/ekstraksi fitur dipanggil pada inferensi.
    _apply_metadata_to_globals(metadata)

    label_meta = None
    if label_meta_path.exists():
        with open(label_meta_path) as f:
            label_meta = json.load(f)

    loaded["metadata"]   = metadata
    loaded["label_meta"] = label_meta
    loaded["models"]     = {}

    rf_path = model_dir / "rf_mfcc.joblib"
    if rf_path.exists():
        loaded["models"]["rf_mfcc"] = joblib.load(rf_path)
    else:
        errors.append("rf_mfcc.joblib")

    try:
        import tensorflow as tf  # noqa: F401
        for key in ("cnn_melspec", "cnn_mfcc", "lstm_melspec", "lstm_mfcc"):
            p = model_dir / f"{key}.keras"
            if not p.exists():
                errors.append(f"{key}.keras")
                continue
            try:
                loaded["models"][key] = _load_keras_model_compat(str(p))
            except RuntimeError as exc:
                errors.append(f"{key}.keras — {exc}")
    except ImportError:
        errors.append("TensorFlow tidak terinstal — model DL tidak dapat dimuat.")

    if errors:
        st.warning(
            "⚠️ File atau model berikut tidak dapat dimuat:\n"
            + "\n".join(f"  • `{e}`" for e in errors)
        )

    return loaded


# ===========================================================================
# PREPROCESSING
# ===========================================================================

def load_audio_bytes(audio_bytes: bytes, ext: str = ".wav") -> np.ndarray:
    """Konversi bytes audio ke array numpy float32 16 kHz mono."""
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        audio = AudioSegment.from_file(tmp_path)
        audio = audio.set_frame_rate(AUDIO_SR).set_channels(1)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        y, _ = librosa.load(wav_io, sr=AUDIO_SR, mono=True)
    finally:
        os.unlink(tmp_path)
    return y.astype(np.float32)


def preprocess_audio(y: np.ndarray) -> dict:
    """Pipeline preprocessing — mengembalikan sinyal per tahap untuk visualisasi."""
    stages = {"raw": y.copy()}

    noise_clip = y[:int(0.5 * AUDIO_SR)]
    y_denoised = nr.reduce_noise(y=y, sr=AUDIO_SR, y_noise=noise_clip)
    stages["denoised"] = y_denoised.copy()

    rms          = np.sqrt(np.mean(y_denoised ** 2))
    current_dBFS = 20 * np.log10(rms) if rms > 0 else -np.inf
    scale        = 10 ** ((-20.0 - current_dBFS) / 20)
    y_normloud   = np.clip(y_denoised * scale, -1.0, 1.0).astype(np.float32)
    stages["normloud"] = y_normloud.copy()

    target_len = int(AUDIO_SR * AUDIO_DURATION)
    y_fixed    = (
        np.pad(y_normloud, (0, target_len - len(y_normloud)))
        if len(y_normloud) < target_len
        else y_normloud[:target_len]
    )
    stages["fixedlen"] = y_fixed.astype(np.float32)

    return stages


def extract_all_features(y: np.ndarray) -> dict:
    """Ekstrak vektor MFCC, matriks MFCC, dan Mel-spectrogram."""
    max_frames = int(np.ceil(int(AUDIO_SR * AUDIO_DURATION) / HOP_LENGTH)) + 1

    def _pad_or_trim(M, target):
        cur = M.shape[1]
        if cur > target:   return M[:, :target]
        elif cur < target: return np.pad(M, ((0, 0), (0, target - cur)), mode="constant")
        return M

    mfcc_mat = librosa.feature.mfcc(
        y=y, sr=AUDIO_SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    ).astype(np.float32)
    mfcc_vec = mfcc_mat.mean(axis=1).astype(np.float32)
    mfcc_mat = _pad_or_trim(mfcc_mat, max_frames)

    S    = librosa.feature.melspectrogram(
        y=y, sr=AUDIO_SR, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmax=FMAX, power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    S_db = _pad_or_trim(S_db, max_frames)

    return {"mfcc_vec": mfcc_vec, "mfcc_mat": mfcc_mat, "melspec": S_db}


# ===========================================================================
# INFERENSI
# ===========================================================================

def predict(model_obj, arch, feature, features, label_meta) -> tuple[str, np.ndarray]:
    """Jalankan inferensi — mengembalikan (predicted_class, proba_array)."""
    if arch == "rf":
        proba    = model_obj.predict_proba(features["mfcc_vec"].reshape(1, -1))[0]
        pred_idx = int(np.argmax(proba))
    else:
        import tensorflow as tf
        arr = features["melspec"] if feature == "melspec" else features["mfcc_mat"]
        x   = (
            arr[np.newaxis, :, :, np.newaxis].astype(np.float32)
            if arch == "cnn"
            else arr.T[np.newaxis, :, :].astype(np.float32)
        )
        proba    = model_obj.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(proba))

    pred_class = (
        label_meta["id2label"][str(pred_idx)] if label_meta else str(pred_idx)
    )
    return pred_class, proba


# ===========================================================================
# VISUALISASI
# ===========================================================================

def _fig_style(fig, ax=None, axes=None):
    fig.patch.set_facecolor(BG_CARD)
    targets = axes if axes else ([ax] if ax else fig.get_axes())
    for a in targets:
        a.set_facecolor(BG_MAIN)
        a.tick_params(colors=CLR_MUTE, labelsize=7)
        for spine in a.spines.values():
            spine.set_edgecolor("#1e293b")
        a.xaxis.label.set_color(CLR_MUTE)
        a.yaxis.label.set_color(CLR_MUTE)
        a.title.set_color(CLR_TEXT)
    return fig


def fig_waveform(y, title="Waveform", color=CLR_CYAN):
    t = np.linspace(0, len(y) / AUDIO_SR, len(y))
    fig, ax = plt.subplots(figsize=(9, 2.2), dpi=110)
    ax.fill_between(t, y, alpha=0.35, color=color)
    ax.plot(t, y, lw=0.6, color=color)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Waktu (s)", fontsize=8)
    ax.set_ylabel("Amplitudo", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    _fig_style(fig, ax)
    fig.tight_layout()
    return fig


def fig_feature_map(matrix, title, y_label="", is_db=True):
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=110)
    img = ax.imshow(matrix, aspect="auto", origin="lower",
                    cmap=CMAP_SPEC, interpolation="bilinear")
    cb = fig.colorbar(img, ax=ax, pad=0.01)
    cb.ax.tick_params(colors=CLR_MUTE, labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=CLR_MUTE, fontsize=6)
    if is_db:
        cb.set_label("dB", color=CLR_MUTE, fontsize=7)
    n_frames = matrix.shape[1]
    ax.set_xticks(np.linspace(0, n_frames - 1, 5))
    ax.set_xticklabels([f"{v:.1f}s" for v in np.linspace(0, AUDIO_DURATION, 5)])
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    _fig_style(fig, ax)
    fig.tight_layout()
    return fig


def fig_rf_architecture(num_classes, feature_dim=N_MFCC):
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=110)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    fig.patch.set_facecolor(BG_CARD); ax.set_facecolor(BG_CARD)
    ax.text(0.1, 2.0, f"MFCC\nVector\n({feature_dim}d)", ha="center", va="center",
            fontsize=7.5, color=CLR_CYAN, fontfamily="monospace",
            bbox=dict(fc=BG_MAIN, ec=CLR_CYAN, lw=1, boxstyle="round,pad=0.4"))
    for i, tx in enumerate([2.5, 5.0, 7.5]):
        ax.add_patch(plt.Circle((tx, 3.0), 0.22, fc=CLR_VIOL, ec="none", alpha=0.9))
        ax.plot([tx-0.22, tx-0.7], [3.0, 2.2], color=CLR_MUTE, lw=0.9)
        ax.add_patch(plt.Circle((tx-0.7, 2.1), 0.16, fc=CLR_CYAN, ec="none", alpha=0.7))
        ax.plot([tx+0.22, tx+0.7], [3.0, 2.2], color=CLR_MUTE, lw=0.9)
        ax.add_patch(plt.Circle((tx+0.7, 2.1), 0.16, fc=CLR_CYAN, ec="none", alpha=0.7))
        for dx in [-1.1, -0.3, 0.3, 1.1]:
            ax.add_patch(plt.Circle((tx+dx, 1.2), 0.12, fc=CLR_AMBR, ec="none", alpha=0.7))
        ax.text(tx, 3.55, f"Tree {i+1}", ha="center", va="bottom",
                fontsize=6.5, color=CLR_MUTE, fontfamily="monospace")
        ax.annotate("", xy=(tx-0.22, 3.0), xytext=(0.42, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color=CLR_MUTE, lw=0.8, alpha=0.6))
    ax.text(9.6, 2.0, "Majority\nVote\n→ Class", ha="center", va="center",
            fontsize=7.5, color=CLR_AMBR, fontfamily="monospace",
            bbox=dict(fc=BG_MAIN, ec=CLR_AMBR, lw=1, boxstyle="round,pad=0.4"))
    for tx in [2.5, 5.0, 7.5]:
        ax.annotate("", xy=(9.15, 2.0), xytext=(tx+0.3, 1.0),
                    arrowprops=dict(arrowstyle="-|>", color=CLR_MUTE, lw=0.7, alpha=0.5))
    ax.set_title("Arsitektur Random Forest (Ensemble 500 Trees)",
                 fontsize=9, color=CLR_TEXT, fontweight="bold", pad=6)
    fig.tight_layout()
    return fig


def fig_cnn_architecture(input_shape, num_classes):
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=110)
    ax.set_xlim(-0.5, 10); ax.set_ylim(-0.5, 4); ax.axis("off")
    fig.patch.set_facecolor(BG_CARD); ax.set_facecolor(BG_CARD)
    layers_info = [
        ("Input\n2D Map",                0.4, CLR_CYAN),
        ("Conv2D\n+BN+ReLU\nFilters:16", 1.9, CLR_VIOL),
        ("MaxPool\n2×2",                 3.1, CLR_MUTE),
        ("Conv2D\n+BN+ReLU\nFilters:32", 4.4, CLR_VIOL),
        ("MaxPool\n2×2",                 5.6, CLR_MUTE),
        ("Conv2D\n+BN+ReLU\nFilters:64", 6.9, CLR_VIOL),
        ("GAP\n+Dropout",                8.1, CLR_AMBR),
        (f"Dense\n{num_classes} cls\nSoftmax", 9.4, CLR_CYAN),
    ]
    prev_x = None
    for label, x, c in layers_info:
        h = 1.8 if "Conv" in label else (1.2 if "Dense" in label else 1.0)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x-0.38, 2.0-h/2), 0.76, h,
            boxstyle="round,pad=0.06", fc=c, ec="none", alpha=0.85, zorder=3
        ))
        ax.text(x, 2.0, label, ha="center", va="center",
                fontsize=6.2, color=BG_MAIN, fontfamily="monospace",
                fontweight="bold", zorder=4)
        if prev_x is not None:
            ax.annotate("", xy=(x-0.38, 2.0), xytext=(prev_x+0.38, 2.0),
                        arrowprops=dict(arrowstyle="-|>", color=CLR_MUTE, lw=0.9), zorder=2)
        prev_x = x
    ax.text(0.4, 0.1, input_shape, ha="center", va="bottom",
            fontsize=6, color=CLR_MUTE, fontfamily="monospace")
    ax.set_title("Arsitektur CNN — Klasifikasi Representasi 2D",
                 fontsize=9, color=CLR_TEXT, fontweight="bold", pad=6)
    fig.tight_layout()
    return fig


def fig_lstm_architecture(input_shape, num_classes):
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=110)
    ax.set_xlim(-0.5, 10); ax.set_ylim(0, 4.5); ax.axis("off")
    fig.patch.set_facecolor(BG_CARD); ax.set_facecolor(BG_CARD)
    for i in range(5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (i*0.38, 0.8), 0.28, 0.55,
            boxstyle="round,pad=0.04", fc=CLR_CYAN, ec="none", alpha=0.6 + i*0.07
        ))
        ax.text(i*0.38+0.14, 1.08, f"t{i+1}", ha="center", va="center",
                fontsize=5.5, color=BG_MAIN, fontfamily="monospace", fontweight="bold")
    ax.text(2.1, 0.55, input_shape, ha="center", va="top",
            fontsize=6, color=CLR_MUTE, fontfamily="monospace")
    for lx, units, alpha in [(3.8, 64, 0.85), (6.0, 32, 0.75)]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (lx-0.55, 0.6), 1.1, 1.8,
            boxstyle="round,pad=0.08", fc=CLR_VIOL, ec="none", alpha=alpha
        ))
        ax.text(lx, 1.5,
                f"LSTM\nLayer {'1' if units==64 else '2'}\nunits={units}",
                ha="center", va="center",
                fontsize=6.8, color=BG_MAIN, fontfamily="monospace", fontweight="bold")
        ax.text(lx, 2.7, "→", ha="center", va="bottom", fontsize=7, color=CLR_AMBR)
    ax.annotate("", xy=(3.25, 1.5), xytext=(1.92, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=CLR_MUTE, lw=0.9))
    ax.annotate("", xy=(5.45, 1.5), xytext=(4.35, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=CLR_MUTE, lw=0.9))
    ax.add_patch(mpatches.FancyBboxPatch(
        (7.6, 0.9), 0.9, 1.2,
        boxstyle="round,pad=0.06", fc=CLR_AMBR, ec="none", alpha=0.8
    ))
    ax.text(8.05, 1.5, "Drop\nout", ha="center", va="center",
            fontsize=6.5, color=BG_MAIN, fontfamily="monospace", fontweight="bold")
    ax.annotate("", xy=(7.6, 1.5), xytext=(6.55, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=CLR_MUTE, lw=0.9))
    ax.add_patch(mpatches.FancyBboxPatch(
        (8.9, 0.9), 0.9, 1.2,
        boxstyle="round,pad=0.06", fc=CLR_CYAN, ec="none", alpha=0.9
    ))
    ax.text(9.35, 1.5, f"Dense\n{num_classes}\ncls", ha="center", va="center",
            fontsize=6.5, color=BG_MAIN, fontfamily="monospace", fontweight="bold")
    ax.annotate("", xy=(8.9, 1.5), xytext=(8.5, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=CLR_MUTE, lw=0.9))
    ax.set_title("Arsitektur LSTM — Klasifikasi Sekuens Temporal",
                 fontsize=9, color=CLR_TEXT, fontweight="bold", pad=6)
    fig.tight_layout()
    return fig


def fig_confidence(proba, classes, pred_class):
    colors = [CLR_CYAN if c == pred_class else CLR_VIOL for c in classes]
    fig = go.Figure(go.Bar(
        x=classes, y=(proba * 100).tolist(),
        marker_color=colors, marker_line_width=0,
        text=[f"{v:.1f}%" for v in proba * 100],
        textposition="outside",
        textfont=dict(size=10, color=CLR_TEXT, family="Space Mono"),
    ))
    fig.update_layout(
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_MAIN,
        font=dict(family="DM Sans", color=CLR_TEXT),
        xaxis=dict(title="Kelas",
                   tickfont=dict(family="Space Mono", size=9, color=CLR_MUTE),
                   gridcolor="#1e293b"),
        yaxis=dict(title="Confidence (%)",
                   tickfont=dict(size=9, color=CLR_MUTE),
                   gridcolor="#1e293b", range=[0, 115]),
        margin=dict(l=10, r=10, t=20, b=10),
        height=280, showlegend=False,
    )
    return fig


# ===========================================================================
# INPUT AUDIO — SUMBER TUNGGAL (MUTUAL EXCLUSIVE)
# ===========================================================================

# Kunci session_state yang digunakan untuk mengelola sumber aktif:
#   "audio_source"   : "upload" | "record" | None
#   "audio_bytes"    : bytes | None
#   "audio_ext"      : str
#   "audio_filename" : str | None   — hanya untuk sumber "upload"

_SRC_UPLOAD = "upload"
_SRC_RECORD = "record"


def _init_audio_state() -> None:
    """
    Inisialisasi session_state audio jika belum ada.
    Default sumber aktif adalah 'upload' — user dapat berganti ke 'record'
    melalui tautan swap yang tersedia di area input.
    """
    defaults = {
        "audio_source":   _SRC_UPLOAD,   # langsung ke mode upload saat pertama buka
        "audio_bytes":    None,
        "audio_ext":      ".wav",
        "audio_filename": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _clear_audio() -> None:
    """Hapus semua state audio — kembali ke kondisi kosong."""
    st.session_state["audio_source"]   = None
    st.session_state["audio_bytes"]    = None
    st.session_state["audio_ext"]      = ".wav"
    st.session_state["audio_filename"] = None


def render_audio_input() -> tuple[bytes | None, str]:
    """
    Render area input audio dengan logika mutual exclusive:
      - Default (pertama buka): langsung tampilkan file uploader.
      - Jika sumber "upload" aktif: tampilkan hanya file uploader.
      - Jika sumber "record" aktif: tampilkan hanya widget rekam.
    Pada setiap kondisi tersedia tautan swap untuk berpindah sumber.

    Mengembalikan (audio_bytes, audio_ext).
    """
    _init_audio_state()
    src = st.session_state["audio_source"]

    # ── Sumber "upload" aktif ───────────────────────────────────────────────
    if src == _SRC_UPLOAD:
        st.markdown(
            '<div class="audio-input-wrap">'
            '<span class="source-badge source-badge-upload">📁 Upload File</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#64748b;font-size:0.82rem;margin:0 0 0.6rem;">'
            'Format yang didukung: WAV · MP3 · FLAC · OGG · M4A</p>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Upload audio",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            label_visibility="collapsed",
            key="file_uploader_widget",
        )

        if uploaded is not None:
            # Simpan ke session_state hanya jika file baru atau berbeda
            if st.session_state["audio_filename"] != uploaded.name:
                st.session_state["audio_bytes"]    = uploaded.read()
                st.session_state["audio_ext"]      = (
                    Path(uploaded.name).suffix.lower() or ".wav"
                )
                st.session_state["audio_filename"] = uploaded.name

            ext = st.session_state["audio_ext"]
            st.audio(
                st.session_state["audio_bytes"],
                format=f"audio/{ext.lstrip('.')}",
            )
            st.caption(
                f"File: `{uploaded.name}` · "
                f"Ukuran: {len(st.session_state['audio_bytes'])/1024:.1f} KB"
            )
        else:
            # File dihapus dari uploader — reset bytes tapi tetap di mode upload
            st.session_state["audio_bytes"]    = None
            st.session_state["audio_filename"] = None

        # Tautan ganti sumber
        st.markdown(
            '<div class="swap-hint">Ingin menggunakan sumber lain? &nbsp;',
            unsafe_allow_html=True,
        )
        if st.button("Rekam langsung →", key="btn_swap_to_record"):
            _clear_audio()
            st.session_state["audio_source"] = _SRC_RECORD
            st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)

        return st.session_state["audio_bytes"], st.session_state["audio_ext"]

    # ── Sumber "record" aktif ───────────────────────────────────────────────
    if src == _SRC_RECORD:
        st.markdown(
            '<div class="audio-input-wrap">'
            '<span class="source-badge source-badge-record">🎙️ Rekam Langsung</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#64748b;font-size:0.82rem;margin:0 0 0.6rem;">'
            'Tekan tombol mikrofon, rekam suara Anda, lalu tekan stop.</p>',
            unsafe_allow_html=True,
        )

        try:
            recorded = st.audio_input("Rekam audio", key="audio_input_widget")
            if recorded is not None:
                raw = recorded.read()
                # Simpan ke session_state hanya jika rekaman berbeda (ukuran berubah)
                if st.session_state["audio_bytes"] is None or len(raw) != len(st.session_state["audio_bytes"]):
                    st.session_state["audio_bytes"] = raw
                    st.session_state["audio_ext"]   = ".wav"

                st.audio(st.session_state["audio_bytes"], format="audio/wav")
                st.caption(
                    f"Rekaman: {len(st.session_state['audio_bytes'])/1024:.1f} KB · WAV mono"
                )
            else:
                st.session_state["audio_bytes"] = None

        except AttributeError:
            st.info(
                "ℹ️ Fitur rekam langsung membutuhkan Streamlit ≥1.31. "
                "Silakan gunakan **Upload File** sebagai alternatif."
            )
            # Otomatis tawarkan ganti ke upload
            if st.button("Beralih ke Upload File →", key="btn_fallback_upload"):
                _clear_audio()
                st.session_state["audio_source"] = _SRC_UPLOAD
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            return None, ".wav"

        # Tautan ganti sumber
        st.markdown(
            '<div class="swap-hint">Ingin menggunakan sumber lain? &nbsp;',
            unsafe_allow_html=True,
        )
        if st.button("Upload file →", key="btn_swap_to_upload"):
            _clear_audio()
            st.session_state["audio_source"] = _SRC_UPLOAD
            st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)

        return st.session_state["audio_bytes"], st.session_state["audio_ext"]

    # Fallback (tidak seharusnya tercapai)
    return None, ".wav"


# ===========================================================================
# PIPELINE INTERAKTIF (5 STEP)
# ===========================================================================

def run_interactive_pipeline(y_raw, variant_name, arch, feature, model_obj, label_meta, metadata):
    """
    Jalankan pipeline analisis secara animatif step-by-step:
      Step 1 — Waveform audio mentah
      Step 2 — Preprocessing
      Step 3 — Ekstraksi fitur
      Step 4 — Ilustrasi arsitektur model
      Step 5 — Prediksi & distribusi confidence
    """
    classes      = label_meta["classes"] if label_meta else [str(i) for i in range(10)]
    num_classes  = len(classes)
    input_shapes = metadata.get("input_shapes", {})

    st.markdown("---")
    st.markdown(
        f'<p class="step-label">Pipeline Analisis — '
        f'<span class="model-badge">{variant_name}</span></p>',
        unsafe_allow_html=True,
    )
    prog = st.progress(0, text="Memulai analisis...")

    # STEP 1
    st.markdown(
        '<div class="step-card">'
        '<div class="step-label">Step 1 / 5</div>'
        '<div class="step-title">🎵 Sinyal Audio Mentah</div>'
        '</div>', unsafe_allow_html=True,
    )
    ph_wf = st.empty()
    prog.progress(10, text="Membaca sinyal audio...")
    time.sleep(0.4)
    ph_wf.pyplot(fig_waveform(y_raw, "Waveform — Audio Asli", CLR_CYAN))
    plt.close("all")

    # STEP 2
    st.markdown(
        '<div class="step-card">'
        '<div class="step-label">Step 2 / 5</div>'
        '<div class="step-title">⚙️ Preprocessing Audio</div>'
        '</div>', unsafe_allow_html=True,
    )
    ph_prep = st.empty()
    col_p1, col_p2, col_p3 = st.columns(3)

    prog.progress(22, text="Melakukan noise reduction...")
    ph_prep.info("🔇 **Noise Reduction** — Memfilter derau latar belakang...")
    time.sleep(0.5)
    stages = preprocess_audio(y_raw)

    with col_p1:
        st.pyplot(fig_waveform(stages["denoised"], "Setelah Noise Reduction", CLR_VIOL))
        plt.close("all")
        st.caption("Profil noise diekstrak dari 0,5 detik pertama, lalu dikurangi secara spektral.")

    prog.progress(35, text="Menormalisasi volume...")
    ph_prep.info("🔊 **Volume Normalization** — Menyamakan level ke −20 dBFS...")
    time.sleep(0.5)

    with col_p2:
        st.pyplot(fig_waveform(stages["normloud"], "Setelah Normalisasi Volume", CLR_AMBR))
        plt.close("all")
        st.caption("Amplitudo diskalakan agar RMS setara −20 dBFS di seluruh sampel.")

    prog.progress(45, text="Menyeragamkan durasi...")
    ph_prep.info("⏱️ **Duration Normalization** — Menyeragamkan ke 5 detik...")
    time.sleep(0.5)

    with col_p3:
        st.pyplot(fig_waveform(stages["fixedlen"], "Setelah Normalisasi Durasi", CLR_CYAN))
        plt.close("all")
        st.caption("Audio dipangkas jika >5 detik, atau diberi silence padding jika <5 detik.")

    ph_prep.success("✅ **Preprocessing selesai** — Audio siap untuk ekstraksi fitur.")
    y_clean = stages["fixedlen"]

    # STEP 3
    st.markdown(
        '<div class="step-card">'
        '<div class="step-label">Step 3 / 5</div>'
        '<div class="step-title">📊 Ekstraksi Fitur</div>'
        '</div>', unsafe_allow_html=True,
    )
    ph_feat = st.empty()
    prog.progress(58, text="Mengekstrak fitur audio...")
    feat_name = "MFCC" if feature == "mfcc" else "Mel-Spectrogram"
    ph_feat.info(f"⚗️ **Ekstraksi** — Menghitung {feat_name}...")
    time.sleep(0.6)
    features = extract_all_features(y_clean)

    if feature == "melspec":
        st.pyplot(fig_feature_map(
            features["melspec"],
            title=f"Mel-Spectrogram ({N_MELS} mel bands × frames)",
            y_label="Mel Band", is_db=True,
        ))
        plt.close("all")
        st.caption(
            f"Mel-spectrogram dihitung dengan n_mels={N_MELS}, n_fft={N_FFT}, "
            f"hop_length={HOP_LENGTH}, fmax={FMAX} Hz, lalu dikonversi ke skala dB."
        )
    else:
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            st.pyplot(fig_feature_map(
                features["mfcc_mat"],
                title=f"Matriks MFCC ({N_MFCC} koefisien × frames)",
                y_label="Koefisien MFCC", is_db=False,
            ))
            plt.close("all")
        with col_f2:
            fig_vec, ax_vec = plt.subplots(figsize=(3.5, 2.6), dpi=110)
            ax_vec.barh(
                range(N_MFCC), features["mfcc_vec"],
                color=[CLR_CYAN if v >= 0 else CLR_AMBR for v in features["mfcc_vec"]],
                height=0.7,
            )
            ax_vec.set_xlabel("Nilai Mean", fontsize=7)
            ax_vec.set_ylabel("Koef.", fontsize=7)
            ax_vec.set_title("Vektor MFCC (mean)", fontsize=8, fontweight="bold")
            _fig_style(fig_vec, ax_vec)
            fig_vec.tight_layout()
            st.pyplot(fig_vec)
            plt.close("all")
        st.caption(
            f"Matriks MFCC ({N_MFCC}×frames) digunakan oleh model deep learning. "
            f"Vektor rata-rata ({N_MFCC}d) digunakan oleh Random Forest."
        )
    ph_feat.success("✅ **Ekstraksi fitur selesai.**")

    # STEP 4
    st.markdown(
        '<div class="step-card">'
        '<div class="step-label">Step 4 / 5</div>'
        '<div class="step-title">🧠 Arsitektur model</div>'
        '</div>', unsafe_allow_html=True,
    )
    ph_arch = st.empty()
    prog.progress(72, text="Memuat ilustrasi arsitektur...")
    ph_arch.info(f"🏗️ Menyusun diagram arsitektur **{variant_name}**...")
    time.sleep(0.5)

    if arch == "rf":
        st.pyplot(fig_rf_architecture(num_classes, feature_dim=N_MFCC))
        st.caption(
            "Random Forest terdiri dari 500 decision tree yang dilatih secara independen "
            "dengan subset fitur acak. Prediksi akhir ditentukan melalui majority voting."
        )
    elif arch == "cnn":
        sk   = "cnn_melspec" if feature == "melspec" else "cnn_mfcc"
        ish  = input_shapes.get(sk, {})
        sstr = (f"({ish.get('n_mels',N_MELS)}, {ish.get('n_frames','T')}, 1)"
                if feature == "melspec"
                else f"({ish.get('n_mfcc',N_MFCC)}, {ish.get('n_frames','T')}, 1)")
        st.pyplot(fig_cnn_architecture(sstr, num_classes))
        st.caption(
            "CNN menggunakan 3 blok Conv2D+BatchNorm+ReLU+MaxPool untuk mengekstrak "
            "pola lokal spasial dari representasi 2D audio, diikuti GlobalAveragePooling "
            "dan Dropout sebelum klasifikasi."
        )
    else:
        sk   = "lstm_melspec" if feature == "melspec" else "lstm_mfcc"
        ish  = input_shapes.get(sk, {})
        sstr = (f"({ish.get('n_frames','T')}, {ish.get('n_mels',N_MELS)})"
                if feature == "melspec"
                else f"({ish.get('n_frames','T')}, {ish.get('n_mfcc_coef',N_MFCC)})")
        st.pyplot(fig_lstm_architecture(sstr, num_classes))
        st.caption(
            "LSTM dua-lapis (stacked) memproses sekuens temporal secara berurutan, "
            "dengan lapisan pertama mengembalikan seluruh sekuens hidden state sebagai "
            "masukan bagi lapisan kedua, sehingga mampu menangkap dependensi temporal "
            "hierarkis pada sinyal vokalisasi."
        )
    plt.close("all")
    ph_arch.success("✅ **Ilustrasi arsitektur ditampilkan.**")

    # STEP 5
    st.markdown(
        '<div class="step-card">'
        '<div class="step-label">Step 5 / 5</div>'
        '<div class="step-title">🎯 Prediksi & Confidence</div>'
        '</div>', unsafe_allow_html=True,
    )
    ph_infer = st.empty()
    prog.progress(88, text="Menjalankan inferensi...")
    ph_infer.info("⚡ Menjalankan model — mohon tunggu...")
    time.sleep(0.4)

    t0 = time.perf_counter()
    pred_class, proba = predict(model_obj, arch, feature, features, label_meta)
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000

    prog.progress(100, text="Analisis selesai!")
    ph_infer.empty()
    time.sleep(0.3)

    top_conf = float(proba.max()) * 100
    st.markdown(
        f'<div class="pred-box">'
        f'<div class="pred-conf">PREDICTED CLASS</div>'
        f'<div class="pred-class">{pred_class.upper()}</div>'
        f'<div class="pred-conf" style="margin-top:0.6rem;">'
        f'Confidence: <strong style="color:#00e5cc">{top_conf:.1f}%</strong>'
        f'&nbsp;|&nbsp;'
        f'Waktu inferensi: <strong style="color:#a78bfa">{infer_ms:.1f} ms</strong>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p class="step-label" style="margin-bottom:0.5rem;">'
        'Distribusi Confidence per Kelas</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_confidence(proba, classes, pred_class), use_container_width=True)

    with st.expander("📋 Tabel confidence lengkap"):
        df_conf = pd.DataFrame({
            "Kelas":          classes,
            "Confidence":     proba,
            "Confidence (%)": [f"{v*100:.2f}%" for v in proba],
        }).sort_values("Confidence", ascending=False)
        df_conf.index = range(1, len(df_conf) + 1)
        st.dataframe(df_conf[["Kelas", "Confidence (%)"]], use_container_width=True)

    return pred_class, proba


# ===========================================================================
# LAYOUT UTAMA
# ===========================================================================

def main():

    # HEADER
    with st.container():
        st.markdown(
            '<div class="hero-title">Javan Sparrow Audio Classifier</div>'
            '<div class="hero-sub">'
            'Dashboard interaktif untuk klasifikasi audio burung Gelatik Jawa'
            '</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:0.7rem;'
            'color:#00e5cc;letter-spacing:0.15em;text-transform:uppercase;">'
            '🤖 Pilih model</p>',
            unsafe_allow_html=True,
        )

        arch_choice = st.radio(
            "Arsitektur",
            options=["Random Forest", "CNN", "LSTM"],
            index=0,
            help="Pilih arsitektur model yang akan digunakan.",
        )
        feature_choice = st.radio(
            "Metode Ekstraksi Fitur",
            options=["MFCC", "Mel-Spectrogram"],
            index=0,
            disabled=(arch_choice == "Random Forest"),
            help="Random Forest hanya mendukung MFCC.",
        )

        if arch_choice == "Random Forest":
            variant_name = "RF + MFCC"
        elif arch_choice == "CNN":
            variant_name = "CNN + MFCC" if feature_choice == "MFCC" else "CNN + MelSpec"
        else:
            variant_name = "LSTM + MFCC" if feature_choice == "MFCC" else "LSTM + MelSpec"

        cfg = VARIANT_OPTIONS[variant_name]

        st.markdown(
            f'<br><div style="text-align:center;">'
            f'<span class="model-badge">{variant_name}</span></div>',
            unsafe_allow_html=True,
        )
        arch_desc = {
            "RF + MFCC":      "Ensemble 500 decision trees · vektor MFCC 40d",
            "CNN + MelSpec":  "3-blok Conv2D · Mel-Spectrogram 2D",
            "CNN + MFCC":     "3-blok Conv2D · matriks MFCC 2D",
            "LSTM + MelSpec": "2-layer LSTM stacked · sekuens dari Mel-Spectrogram",
            "LSTM + MFCC":    "2-layer LSTM stacked · sekuens dari matriks MFCC",
        }
        st.caption(arch_desc[variant_name])
        st.markdown("---")
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:0.6rem;'
            'color:#64748b;text-align:center;">'
            '© Pasha Rakha Paruntung · Universitas Atma Jaya Yogyakarta</p>',
            unsafe_allow_html=True,
        )

    # PEMUATAN MODEL
    with st.spinner("Memuat model... (proses ini hanya terjadi sekali per sesi)"):
        artifacts = load_all_models(MODEL_DIR)

    if artifacts is None:
        st.stop()

    model_obj  = artifacts["models"].get(cfg["model_key"])
    label_meta = artifacts["label_meta"]
    metadata   = artifacts["metadata"]

    if model_obj is None:
        st.error(
            f"❌ model `{cfg['model_key']}` belum dapat dimuat. "
            "Periksa konstanta `MODEL_DIR` atau pastikan semua "
            "file `.keras` / `.joblib` tersedia."
        )
        st.stop()

    # INPUT & INFO
    col_main, col_info = st.columns([3, 1], gap="large")

    with col_main:
        audio_bytes, audio_ext = render_audio_input()

    with col_info:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="step-card">'
            '<div class="step-label">model Aktif</div>'
            f'<div class="step-title">{variant_name}</div><br>'
            f'<div style="font-size:0.78rem;color:#64748b;">'
            f'<b style="color:#e2e8f0">Arsitektur :</b> {cfg["arch"].upper()}<br>'
            f'<b style="color:#e2e8f0">Fitur       :</b> {cfg["feature"].upper()}<br>'
            f'<b style="color:#e2e8f0">Kelas       :</b> '
            f'{len(label_meta["classes"]) if label_meta else "N/A"}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if label_meta:
            classes_html = "".join([
                f'<div style="font-family:\'Space Mono\',monospace;'
                f'font-size:0.72rem;color:#ffffff;padding:2px 0;">› {c}</div>'
                for c in label_meta["classes"]
            ])
            st.markdown(
                f'<div class="step-card">'
                f'<div class="step-label">Daftar Kelas</div>'
                f'{classes_html}</div>',
                unsafe_allow_html=True,
            )

    # TOMBOL KLASIFIKASI
    st.markdown("<br>", unsafe_allow_html=True)
    classify_btn = st.button(
        "Mulai Klasifikasi Audio  ▶",
        disabled=(audio_bytes is None),
        type="primary",
    )

    # EKSEKUSI PIPELINE
    if classify_btn and audio_bytes:
        try:
            with st.spinner("Menguraikan sinyal audio..."):
                y_raw = load_audio_bytes(audio_bytes, ext=audio_ext)
                if len(y_raw) < int(AUDIO_SR * 0.5):
                    st.error("⚠️ Audio terlalu pendek (minimum 0,5 detik). Coba rekam ulang.")
                    st.stop()

            run_interactive_pipeline(
                y_raw        = y_raw,
                variant_name = variant_name,
                arch         = cfg["arch"],
                feature      = cfg["feature"],
                model_obj    = model_obj,
                label_meta   = label_meta,
                metadata     = metadata,
            )
        except Exception as exc:
            st.error(f"❌ Terjadi kesalahan saat memproses audio:\n\n`{exc}`")
            with st.expander("Detail error"):
                import traceback
                st.code(traceback.format_exc(), language="python")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    main()