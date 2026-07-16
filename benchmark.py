import os
import gc
import time
import re
import warnings

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jiwer
from num2words import num2words
from datasets import load_dataset, Audio
import whisper
import pynvml
import psutil

# NVML başlat (GPU donanım seviyesi metrikler için)
try:
    pynvml.nvmlInit()
    HAS_NVML = True
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    HAS_NVML = False

# Kendi modelimizden importlar
from core import STTPipeline
from config import STTConfig, BANKING_INITIAL_PROMPT
from models import TranscriptionResult

# Uyarıları gizle
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"

# Özel cache dizini (Benchmark bitince silinmesi kolay olsun diye)
CACHE_DIR = "./benchmark_cache"


def normalize_text(text: str) -> str:
    """Türkçe metin normalizasyonu: Küçük harf, noktalama temizliği ve sayıların okunuşa çevrilmesi."""
    if not text:
        return ""

    # Küçük harfe çevir (Türkçe karakterleri dikkate alarak basit çevrim)
    text = text.replace("I", "ı").replace("İ", "i").lower()

    # Noktalama işaretlerini temizle
    text = re.sub(r"[^\w\s]", "", text)

    # Sayıları metne çevir (num2words)
    def replace_num(match):
        num_str = match.group()
        try:
            return num2words(int(num_str), lang="tr")
        except:
            return num_str

    text = re.sub(r"\b\d+\b", replace_num, text)

    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_vram_usage():
    """Donanım seviyesinde (NVML) veya PyTorch üzerinden GPU VRAM kullanımını MB cinsinden döndürür."""
    if HAS_NVML:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            return info.used / (1024 * 1024)
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0


def get_system_usage():
    """CPU ve RAM kullanımını anlık olarak döndürür."""
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().used / (1024 * 1024)
    return cpu, ram


def clear_vram():
    """VRAM'i ve RAM'i temizler."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    time.sleep(2)  # Sistemin rahatlaması için kısa bir bekleme


def run_benchmark(num_samples=None):
    print("=== STT Benchmark Başlatılıyor ===")
    print(
        f"HuggingFace üzerinden Google FLEURS (tr_tr) indiriliyor (Cache: {CACHE_DIR})..."
    )

    # Veri setini yükle
    try:
        split_str = "test" if num_samples is None else f"test[:{num_samples}]"
        dataset = load_dataset(
            "google/fleurs",
            "tr_tr",
            split=split_str,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        num_samples = len(dataset)
    except Exception as e:
        print(
            f"Veri seti yüklenirken hata oluştu. HuggingFace token girilmiş mi? Hata: {e}"
        )
        return

    results = []

    # ---------------------------------------------------------
    # 1. Aşama: Bizim Sistemimiz (Faster-Whisper tabanlı STTPipeline)
    # ---------------------------------------------------------
    print("\n[1/2] STTPipeline (Faster-Whisper Tabanlı) Yükleniyor...")
    clear_vram()
    config = STTConfig(
        whisper_model="large-v3",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="int8_float16" if torch.cuda.is_available() else "int8",
        vad_filter=True,
        beam_size=5,
        use_ollama=False,
        vad_min_silence_ms=250,
        temperature=0.0,
        condition_on_previous_text=False,
    )

    pipeline_results = []

    def on_result(res: TranscriptionResult):
        pipeline_results.append(res.text)

    pipeline = STTPipeline(config=config, on_result=on_result)

    # Baseline cpu metric
    psutil.cpu_percent(interval=0.1)

    try:
        for idx, item in enumerate(dataset):
            audio_array = item["audio"]["array"].astype(np.float32)
            duration = len(audio_array) / 16000.0
            reference_text = item["transcription"]
            norm_ref = normalize_text(reference_text)

            pipeline_results.clear()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()

            segments, _ = pipeline._model.transcribe(
                audio_array,
                language=config.language,
                beam_size=config.beam_size,
                vad_filter=config.vad_filter,
                initial_prompt=config.initial_prompt,
                temperature=config.temperature,
                condition_on_previous_text=config.condition_on_previous_text,
            )

            # Halüsinasyon filtresinden geçir
            valid_segments, confidence = pipeline.halucination_filter.filter_segments(
                segments
            )
            pred_text = " ".join([seg.text for seg in valid_segments]).strip()

            from models import TranscriptionStatus

            res_obj = pipeline.halucination_filter.build_result(
                pred_text, valid_segments, confidence, 0
            )

            if res_obj.status in [
                TranscriptionStatus.REJECTED,
                TranscriptionStatus.SILENCE,
            ]:
                pred_text = ""
            elif pred_text and config.use_ollama:
                pred_text = pipeline._clean_filler_words(pred_text)

            end_time = time.time()
            process_time = end_time - start_time
            norm_pred = normalize_text(pred_text)

            wer = jiwer.wer(norm_ref, norm_pred) if norm_ref else 0
            cer = jiwer.cer(norm_ref, norm_pred) if norm_ref else 0
            rtf = process_time / duration

            vram_peak = get_vram_usage()
            cpu_usage, ram_usage = get_system_usage()

            results.append(
                {
                    "Ses_ID": idx,
                    "Model": "Faster Whisper (Bizim Sistem)",
                    "Duration (s)": duration,
                    "Process Time (s)": process_time,
                    "WER (%)": wer * 100,
                    "CER (%)": cer * 100,
                    "RTF": rtf,
                    "VRAM Peak (MB)": vram_peak,
                    "CPU Usage (%)": cpu_usage,
                    "RAM Usage (MB)": ram_usage,
                    "Ref": norm_ref,
                    "Pred": norm_pred,
                }
            )

            if (idx + 1) % 10 == 0:
                print(f"  -> Faster-Whisper: {idx + 1}/{num_samples} tamamlandı...")
    except KeyboardInterrupt:
        print(
            "\n[UYARI] Benchmark yarıda kesildi! O ana kadar işlenen verilerle devam ediliyor..."
        )

    print("STTPipeline bellekten siliniyor...")
    del pipeline
    clear_vram()

    # ---------------------------------------------------------
    # 2. Aşama: OpenAI Whisper (Vanilla)
    # ---------------------------------------------------------
    print("\n[2/2] OpenAI Whisper (Vanilla) Yükleniyor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("large-v3", device=device)

    print(f"OpenAI Whisper ile {num_samples} adet ses işleniyor...")
    try:
        for idx, item in enumerate(dataset):
            audio_array = item["audio"]["array"].astype(np.float32)
            duration = len(audio_array) / 16000.0
            reference_text = item["transcription"]
            norm_ref = normalize_text(reference_text)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start_time = time.time()

            # Normal whisper transcribe işlemi - faster_whisper configleri ile uyumlu
            res = whisper_model.transcribe(
                audio_array,
                language="tr",
                beam_size=config.beam_size,
                initial_prompt=config.initial_prompt,
                temperature=config.temperature,
                condition_on_previous_text=config.condition_on_previous_text,
                fp16=(device == "cuda"),
            )
            pred_text = res["text"]

            end_time = time.time()
            process_time = end_time - start_time
            norm_pred = normalize_text(pred_text)

            wer = jiwer.wer(norm_ref, norm_pred) if norm_ref else 0
            cer = jiwer.cer(norm_ref, norm_pred) if norm_ref else 0
            rtf = process_time / duration

            vram_peak = get_vram_usage()
            cpu_usage, ram_usage = get_system_usage()

            results.append(
                {
                    "Ses_ID": idx,
                    "Model": "OpenAI Whisper (Vanilla)",
                    "Duration (s)": duration,
                    "Process Time (s)": process_time,
                    "WER (%)": wer * 100,
                    "CER (%)": cer * 100,
                    "RTF": rtf,
                    "VRAM Peak (MB)": vram_peak,
                    "CPU Usage (%)": cpu_usage,
                    "RAM Usage (MB)": ram_usage,
                    "Ref": norm_ref,
                    "Pred": norm_pred,
                }
            )

            if (idx + 1) % 10 == 0:
                print(f"  -> Vanilla Whisper: {idx + 1}/{num_samples} tamamlandı...")
    except KeyboardInterrupt:
        print(
            "\n[UYARI] Benchmark kullanıcı tarafından yarıda kesildi! Mevcut verilerle devam ediliyor..."
        )

    print("OpenAI Whisper bellekten siliniyor...")
    del whisper_model
    clear_vram()

    # ---------------------------------------------------------
    # 3. Sonuçların Analizi ve Görselleştirme
    # ---------------------------------------------------------
    df = pd.DataFrame(results)

    # Ortalamaları hesapla
    summary = (
        df.groupby("Model")
        .agg(
            {
                "WER (%)": ["mean", "std"],
                "CER (%)": ["mean", "std"],
                "RTF": ["mean", "std"],
                "Process Time (s)": ["mean", "max"],
                "VRAM Peak (MB)": ["mean", "max"],
                "CPU Usage (%)": "mean",
                "RAM Usage (MB)": "mean",
            }
        )
        .round(3)
    )

    # Çoklu sütun indekslerini düzleştir
    summary.columns = [
        "_".join(col) if isinstance(col, tuple) else col
        for col in summary.columns.values
    ]
    summary = summary.reset_index()

    print("\n====== BENCHMARK ÖZET TABLOSU ======")
    from tabulate import tabulate

    print(tabulate(summary, headers="keys", tablefmt="grid"))

    # CSV olarak kaydet
    df.to_csv("benchmark_raw_results.csv", index=False)
    summary.to_csv("benchmark_summary.csv", index=False)

    # Görselleştirme (Araştırma Standardında)
    print("\nAraştırma kalitesinde grafikler ve tablolar çiziliyor...")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(
        "Faster-Whisper vs Vanilla Whisper STT Performans ve Donanım Analizi",
        fontsize=24,
        fontweight="bold",
        y=0.97,
    )

    # Grid spesifikasyonları: 3 satır x 3 sütun
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

    palette = {
        "Faster Whisper (Bizim Sistem)": "#2E86AB",
        "OpenAI Whisper (Vanilla)": "#D64045",
    }

    # 1. WER (Word Error Rate)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(
        data=df,
        x="Model",
        y="WER (%)",
        ax=ax1,
        errorbar="sd",
        capsize=0.1,
        palette=palette,
    )
    ax1.set_title("Ortalama WER (%) (Düşük Daha İyi)", fontweight="bold")
    ax1.set_ylabel("WER (%)")

    # 2. Ortalama İşlem Süresi
    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(
        data=df,
        x="Model",
        y="Process Time (s)",
        ax=ax2,
        errorbar="sd",
        capsize=0.1,
        palette=palette,
    )
    ax2.set_title("Ortalama İşlem Süresi (Düşük Daha İyi)", fontweight="bold")
    ax2.set_ylabel("Süre (sn)")

    # 3. Ortalama RTF
    ax3 = fig.add_subplot(gs[0, 2])
    sns.barplot(
        data=df, x="Model", y="RTF", ax=ax3, errorbar="sd", capsize=0.1, palette=palette
    )
    ax3.set_title("Ortalama Real-Time Factor (RTF)", fontweight="bold")
    ax3.set_ylabel("RTF")

    # 4. Süreç Uzadıkça İşlem Süresindeki Değişim (Performans Degredasyonu)
    ax4 = fig.add_subplot(gs[1, 0:2])
    sns.lineplot(
        data=df,
        x="Ses_ID",
        y="Process Time (s)",
        hue="Model",
        marker="o",
        ax=ax4,
        palette=palette,
    )
    ax4.set_title(
        "Zamanla Performans Değişimi: İşlem Süresi (Süreç Uzadıkça Degredasyon)",
        fontweight="bold",
    )
    ax4.set_xlabel("İşlenen Ses Sırası (İlerleyiş)")
    ax4.set_ylabel("İşlem Süresi (sn)")

    # 5. CPU Kullanımı Yük Altında Nasıl Değişiyor
    ax5 = fig.add_subplot(gs[1, 2])
    sns.lineplot(
        data=df,
        x="Ses_ID",
        y="CPU Usage (%)",
        hue="Model",
        marker="o",
        ax=ax5,
        palette=palette,
    )
    ax5.set_title("İşlem Boyunca CPU Kullanımı (Yük Altında)", fontweight="bold")
    ax5.set_xlabel("İşlenen Ses Sırası")
    ax5.set_ylabel("CPU Kullanımı (%)")

    # 6. VRAM Yük Altında Değişimi
    ax6 = fig.add_subplot(gs[2, 0:2])
    sns.lineplot(
        data=df,
        x="Ses_ID",
        y="VRAM Peak (MB)",
        hue="Model",
        marker="s",
        ax=ax6,
        palette=palette,
    )
    ax6.set_title(
        "Süreç Uzadıkça VRAM (GPU Bellek) Tüketimi Değişimi", fontweight="bold"
    )
    ax6.set_xlabel("İşlenen Ses Sırası")
    ax6.set_ylabel("VRAM Peak (MB)")

    # 7. Bilgi & Tablo Paneli
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")

    # Matplotlib içine görsel tablo ekleyelim
    headers = ["Metrik", "Faster Whisper", "Vanilla Whisper"]

    fw_df = df[df["Model"] == "Faster Whisper (Bizim Sistem)"]
    vw_df = df[df["Model"] == "OpenAI Whisper (Vanilla)"]

    metrics = [
        (
            "Ortalama WER",
            f"{fw_df['WER (%)'].mean():.2f}%",
            f"{vw_df['WER (%)'].mean():.2f}%" if not vw_df.empty else "-",
        ),
        (
            "Ortalama CER",
            f"{fw_df['CER (%)'].mean():.2f}%",
            f"{vw_df['CER (%)'].mean():.2f}%" if not vw_df.empty else "-",
        ),
        (
            "Ortalama Süre",
            f"{fw_df['Process Time (s)'].mean():.2f}s",
            f"{vw_df['Process Time (s)'].mean():.2f}s" if not vw_df.empty else "-",
        ),
        (
            "Maks İşlem Süresi",
            f"{fw_df['Process Time (s)'].max():.2f}s",
            f"{vw_df['Process Time (s)'].max():.2f}s" if not vw_df.empty else "-",
        ),
        (
            "Ortalama RTF",
            f"{fw_df['RTF'].mean():.2f}",
            f"{vw_df['RTF'].mean():.2f}" if not vw_df.empty else "-",
        ),
        (
            "Maks VRAM Peak",
            f"{fw_df['VRAM Peak (MB)'].max():.0f} MB",
            f"{vw_df['VRAM Peak (MB)'].max():.0f} MB" if not vw_df.empty else "-",
        ),
        (
            "Ortalama CPU",
            f"{fw_df['CPU Usage (%)'].mean():.1f}%",
            f"{vw_df['CPU Usage (%)'].mean():.1f}%" if not vw_df.empty else "-",
        ),
        (
            "Ortalama RAM",
            f"{fw_df['RAM Usage (MB)'].mean():.0f} MB",
            f"{vw_df['RAM Usage (MB)'].mean():.0f} MB" if not vw_df.empty else "-",
        ),
    ]

    table = ax7.table(
        cellText=metrics, colLabels=headers, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Tablo hücre stilleri
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#40466e")
        else:
            if col == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f1f1f2")

    ax7.set_title("Özet Karşılaştırma Raporu (Tablo)", fontweight="bold")

    plt.tight_layout()
    fig.subplots_adjust(top=0.94)  # Suptitle için üst boşluk
    plt.savefig("benchmark_charts_detailed.png", dpi=300, bbox_inches="tight")
    print(
        "\nGrafikler ve tablo 'benchmark_charts_detailed.png' olarak başarıyla kaydedildi."
    )
    print("İşlem tamamlandı!")


if __name__ == "__main__":
    run_benchmark()
