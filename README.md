# AudioSense — Developer Guide

> Peta kode `app.py`: bagian mana mengatur bagian mana dari UI

---

## Daftar Isi

1. [Struktur File](#1-struktur-file)
2. [Peta Kode → UI](#2-peta-kode--ui)
3. [Cara Mengubah Bagian-Bagian Tertentu](#3-cara-mengubah-bagian-bagian-tertentu)
4. [Alur Data End-to-End](#4-alur-data-end-to-end)
5. [Checklist Sebelum Deployment](#5-checklist-sebelum-deployment)

---

## 1. Struktur File

```
project/
├── app.py                    ← Satu-satunya file yang perlu disentuh
├── requirements.txt
├── Models/                   ← Diatur oleh konstanta MODEL_DIR di app.py
│   ├── rf_mfcc.joblib
│   ├── cnn_melspec.keras
│   ├── cnn_mfcc.keras
│   ├── lstm_melspec.keras
│   ├── lstm_mfcc.keras
│   └── model_metadata.json
└── Meta/
    └── label_meta.json       ← Satu level di atas Models/
```

---

## 2. Peta Kode → UI

### 2.1 Konfigurasi Developer (tidak menghasilkan UI secara langsung)

| Konstanta / Blok                                                                | Lokasi di `app.py`                    | Efek pada UI / Sistem                                                                  |
| ------------------------------------------------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------- |
| `MODEL_DIR`                                                                     | Blok `# [DEV] KONFIGURASI PATH MODEL` | Folder model yang dimuat. Ubah ini saat pindah environment.                            |
| `AUDIO_SR`, `AUDIO_DURATION`, `N_MFCC`, `N_MELS`, `HOP_LENGTH`, `N_FFT`, `FMAX` | Blok `# [DEV] KONSTANTA FITUR`        | Harus identik dengan nilai training. Memengaruhi dimensi input model.                  |
| `VARIANT_OPTIONS`                                                               | Blok `# [DEV] REGISTRI VARIAN MODEL`  | Daftar model di sidebar. Tambah/hapus entri untuk mengaktifkan/menonaktifkan varian.   |
| `BG_MAIN`, `CLR_CYAN`, `CLR_VIOL`, `CLR_AMBR`, `CMAP_SPEC`                      | Blok `# [DEV] PALET WARNA TEMA`       | Warna semua visualisasi matplotlib. **Harus sinkron** dengan `--accent-*` di blok CSS. |

---

### 2.2 CSS → Kelas HTML yang Dipanggil dari Python

| Kelas CSS      | Dipanggil di fungsi                        | Tampil sebagai                                        |
| -------------- | ------------------------------------------ | ----------------------------------------------------- |
| `.hero-title`  | `main()` — bagian Header                   | Judul besar "AudioSense" dengan gradient              |
| `.hero-sub`    | `main()` — bagian Header                   | Subjudul "Speech Classification System…"              |
| `.step-card`   | `run_interactive_pipeline()` — setiap step | Kartu abu gelap dengan garis vertikal di kiri         |
| `.step-label`  | Di dalam `.step-card`                      | Teks kecil all-caps bertanda step (mis. "Step 1 / 5") |
| `.step-title`  | Di dalam `.step-card`                      | Teks judul step yang lebih besar                      |
| `.model-badge` | Header pipeline & sidebar                  | Pill badge hijau toska bertuliskan nama varian        |
| `.pred-box`    | Step 5 pipeline                            | Kotak besar hasil prediksi akhir                      |
| `.pred-class`  | Di dalam `.pred-box`                       | Nama kelas prediksi huruf kapital besar               |
| `.pred-conf`   | Di dalam `.pred-box`                       | Baris confidence % dan waktu inferensi                |

> **Aturan sinkronisasi warna:** setiap mengubah `CLR_CYAN / CLR_VIOL / CLR_AMBR`
> di blok Python, ubah juga `--accent-1 / --accent-3 / --accent-2` di blok `<style>`.

---

### 2.3 Sidebar

| Kode di `main()`                          | Elemen UI yang Dihasilkan                              |
| ----------------------------------------- | ------------------------------------------------------ |
| `st.radio("Arsitektur", ...)`             | Tiga pilihan radio: Random Forest / CNN / BiLSTM       |
| `st.radio("Metode Ekstraksi Fitur", ...)` | Dua pilihan: MFCC / Mel-Spectrogram (disabled jika RF) |
| Logika `if arch_choice == ...`            | Resolusi `variant_name` dari dua radio                 |
| `st.markdown(...model-badge...)`          | Pill badge nama varian yang aktif                      |
| `st.caption(arch_desc[variant_name])`     | Teks deskripsi singkat di bawah badge                  |

---

### 2.4 Panel Informasi Model (kolom kanan halaman utama)

| Kode di `main()`                              | Elemen UI yang Dihasilkan                                 |
| --------------------------------------------- | --------------------------------------------------------- |
| `st.markdown(...step-card... Model Aktif...)` | Kartu: nama varian, arsitektur, jenis fitur, jumlah kelas |
| `for c in label_meta["classes"]`              | Daftar kelas di bawah kartu "Daftar Kelas"                |

---

### 2.5 Area Input Audio (kolom kiri halaman utama)

| Kode di `main()`                                   | Elemen UI yang Dihasilkan                              |
| -------------------------------------------------- | ------------------------------------------------------ |
| `st.tabs(["📁 Upload File", "🎙️ Rekam Langsung"])` | Dua tab input                                          |
| `st.file_uploader(...)`                            | Area drag-and-drop di tab pertama                      |
| `st.audio_input(...)`                              | Widget rekam mikrofon di tab kedua (Streamlit >= 1.31) |
| `st.audio(audio_bytes, ...)`                       | Pemutar pratinjau setelah file diunggah                |

---

### 2.6 Pipeline Interaktif — `run_interactive_pipeline()`

| Step          | Fungsi yang Dipanggil                                           | Elemen UI yang Dihasilkan                        |
| ------------- | --------------------------------------------------------------- | ------------------------------------------------ |
| **1**         | `fig_waveform(y_raw, ...)`                                      | Plot waveform sinyal audio asli                  |
| **2a**        | `fig_waveform(stages["denoised"], ...)`                         | Waveform setelah noise reduction (violet)        |
| **2b**        | `fig_waveform(stages["normloud"], ...)`                         | Waveform setelah normalisasi volume (amber)      |
| **2c**        | `fig_waveform(stages["fixedlen"], ...)`                         | Waveform setelah normalisasi durasi (cyan)       |
| **3 MelSpec** | `fig_feature_map(features["melspec"], ...)`                     | Heatmap Mel-Spectrogram                          |
| **3 MFCC**    | `fig_feature_map(features["mfcc_mat"], ...)` + `barh(mfcc_vec)` | Heatmap matriks MFCC + bar chart vektor mean     |
| **4 RF**      | `fig_rf_architecture(...)`                                      | Diagram pohon keputusan ensemble                 |
| **4 CNN**     | `fig_cnn_architecture(...)`                                     | Diagram blok konvolusi berurutan                 |
| **4 LSTM**    | `fig_lstm_architecture(...)`                                    | Diagram sel BiLSTM dua lapis                     |
| **5**         | `fig_confidence(proba, ...)`                                    | Bar chart Plotly interaktif confidence per kelas |
| **5**         | `st.markdown(...pred-box...)`                                   | Kotak besar nama kelas dan confidence            |
| **5**         | `st.expander("📋 Tabel...")`                                    | Tabel detail confidence yang dapat dilipat       |
| **5**         | `st.balloons()` jika confidence >= 90%                          | Animasi balon                                    |

---

### 2.7 Progress Bar — Nilai dan Momen

| Nilai    | Momen                                                |
| -------- | ---------------------------------------------------- |
| 0 → 10   | Mulai membaca sinyal                                 |
| 10 → 22  | Selesai waveform, mulai noise reduction              |
| 22 → 35  | Selesai noise reduction, mulai normalisasi volume    |
| 35 → 45  | Selesai normalisasi volume, mulai normalisasi durasi |
| 45 → 58  | Selesai preprocessing, mulai ekstraksi fitur         |
| 58 → 72  | Selesai ekstraksi fitur, mulai diagram arsitektur    |
| 72 → 88  | Selesai diagram, mulai inferensi                     |
| 88 → 100 | Selesai inferensi, tampilkan hasil                   |

---

## 3. Cara Mengubah Bagian-Bagian Tertentu

### Mengganti path model

```python
# Cari baris ini:
MODEL_DIR = Path("./Models")

# Ganti sesuai environment:
MODEL_DIR = Path("/home/user/skripsi/Models")
```

### Menambah atau menonaktifkan varian model

```python
VARIANT_OPTIONS = {
    "RF + MFCC": {"model_key": "rf_mfcc", "arch": "rf", "feature": "mfcc"},
    # Komentari untuk menonaktifkan:
    # "CNN + MelSpec": {"model_key": "cnn_melspec", "arch": "cnn", "feature": "melspec"},
}
```

Perubahan ini otomatis memperbarui pilihan radio di sidebar.

### Mengubah judul dan subjudul halaman

```python
# Di fungsi main(), bagian # HEADER:
st.markdown(
    '<div class="hero-title">AudioSense</div>'        # ubah judul
    '<div class="hero-sub">Speech Classification...</div>',  # ubah subjudul
    ...
)

# Sekaligus ubah juga di st.set_page_config():
st.set_page_config(page_title="AudioSense — Speech Classifier", ...)
```

### Mengubah warna tema

1. Ubah konstanta Python:
   ```python
   CLR_CYAN = "#00e5cc"   # warna primer
   CLR_VIOL = "#a78bfa"   # warna sekunder
   CLR_AMBR = "#f97316"   # warna aksen
   ```
2. **Wajib** sinkronkan dengan CSS:
   ```css
   :root {
     --accent-1: #00e5cc; /* = CLR_CYAN */
     --accent-2: #f97316; /* = CLR_AMBR */
     --accent-3: #a78bfa; /* = CLR_VIOL */
   }
   ```

### Mengubah teks deskripsi varian di sidebar

```python
# Cari dict arch_desc di main():
arch_desc = {
    "RF + MFCC": "Ensemble 500 decision trees · vektor MFCC 40d",  # ubah teks
    ...
}
```

### Mengubah caption teknis di setiap step

Setiap `st.caption(...)` di dalam `run_interactive_pipeline()` berada tepat
di bawah visualisasi yang bersesuaian. Cari komentar `# STEP N` di atasnya.

### Mengatur kecepatan animasi

```python
# Setiap time.sleep() di run_interactive_pipeline() mengontrol jeda (dalam detik):
time.sleep(0.5)   # naikkan untuk memperlambat, turunkan untuk mempercepat
```

---

## 4. Alur Data End-to-End

```
[User: upload file / rekam mikrofon]
        |
        v
load_audio_bytes()
  - AudioSegment.from_file()   (pydub: terima format apa pun)
  - librosa.load()             (float32 mono 16 kHz)
        |
        v
preprocess_audio()
  - nr.reduce_noise()          (noise reduction spektral)
  - RMS scaling                (volume normalization -20 dBFS)
  - pad / trim                 (duration normalization 5 detik)
  => dict stages untuk visualisasi Step 2
        |
        v
extract_all_features()
  - librosa.feature.mfcc()    => mfcc_vec (40d)    untuk RF
                               => mfcc_mat (40xT)  untuk CNN/LSTM + MFCC
  - librosa.feature.melspectrogram()
  - librosa.power_to_db()     => S_db (128xT)      untuk CNN/LSTM + MelSpec
        |
        v
predict()
  RF   : model.predict_proba(mfcc_vec.reshape(1,-1))
  CNN  : model.predict(arr[newaxis,:,:,newaxis])      # shape (1,H,W,1)
  LSTM : model.predict(arr.T[newaxis,:,:])            # shape (1,T,F)
        |
        v
[Output: pred_class (str), proba (np.ndarray per kelas)]
  - st.markdown(.pred-box.)    => kotak nama kelas + confidence
  - fig_confidence()           => bar chart Plotly
  - st.expander()              => tabel detail
```

---

## 5. Checklist Sebelum Deployment

- [ ] `MODEL_DIR` menunjuk ke folder yang benar
- [ ] Semua file model tersedia: `rf_mfcc.joblib`, `cnn_melspec.keras`, `cnn_mfcc.keras`, `lstm_melspec.keras`, `lstm_mfcc.keras`, `model_metadata.json`
- [ ] `Meta/label_meta.json` berada satu level di atas `Models/`
- [ ] Semua konstanta fitur identik dengan nilai saat training
- [ ] Versi TensorFlow/Keras di environment deployment kompatibel dengan versi saat training
- [ ] `pip install -r requirements.txt` berhasil tanpa error
- [ ] `streamlit run app.py` berjalan dan model dimuat tanpa warning kritis
