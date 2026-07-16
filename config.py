"""
Banking-grade STT yapılandırması.

Tüm pipeline parametreleri, güven eşikleri ve Türkçe halüsinasyon
kalıpları burada merkezi olarak tanımlanır.
"""

import re
from dataclasses import dataclass


# ─── Türkçe Whisper Halüsinasyon Kalıpları ────────────────────────────────────
# Whisper'ın eğitim verisindeki Türkçe altyazı dosyalarından ve YouTube
# içeriklerinden kaynaklanan bilinen sahte çıktılar.
# Sessizlikte veya gürültüde model bu kalıpları "uydurur".
TURKISH_HALLUCINATION_PATTERNS: list[re.Pattern] = [
    # Altyazı meta verileri
    re.compile(r"\baltyazı\b", re.IGNORECASE),
    re.compile(r"\bM\.K\.?\b"),
    re.compile(r"\bsubtitle\b", re.IGNORECASE),
    # Çizgi film / müzik referansları
    re.compile(r"\bçizgi\s*film\b", re.IGNORECASE),
    re.compile(r"\bmüzik\s*çalıyor\b", re.IGNORECASE),
    re.compile(r"\bmüziği\b", re.IGNORECASE),
    # YouTube / sosyal medya kalıpları
    re.compile(r"\babone\s*ol\b", re.IGNORECASE),
    re.compile(r"\bbeğen\b", re.IGNORECASE),
    re.compile(r"\biyi\s*seyirler\b", re.IGNORECASE),
    re.compile(r"\byoutube\b", re.IGNORECASE),
    re.compile(r"\bwww\b", re.IGNORECASE),
    re.compile(r"\.com\b", re.IGNORECASE),
    # Tek başına "teşekkürler" (cümle içinde olursa sorun yok)
    re.compile(r"^\s*teşekkürler\.?\s*$", re.IGNORECASE),
    # "İzlediğiniz için teşekkür" kalıbı (video sonu halüsinasyonu)
    re.compile(r"izlediğiniz\s*için\s*teşekkür", re.IGNORECASE),
    # Müzik sembolleri
    re.compile(r"[♪🎵🎶]"),
]

# Bankacılık alanına özgü ön-prompt.
# Whisper'ı finansal Türkçe kelime dağarcığına yönlendirir,
# altyazı/çizgi film halüsinasyonlarını bastırır.
BANKING_INITIAL_PROMPT = (
    "Vadeli hesap, vadesiz hesap, hesap numarası, IBAN, havale, EFT, "
    "kredi kartı, mevduat, faiz oranı, Türk Lirası, TL, hesap bakiyesi, "
    "para transferi, gönder, transfer et, yatır, çek, öde, sorgula, "
    "ödeme, taksit, döviz kuru, müşteri, bakiye, vadeli, vadesiz, FAST."
)


@dataclass
class STTConfig:
    """Bankacılık düzeyinde STT pipeline yapılandırması."""

    # ── Model ──────────────────────────────────────────────────
    whisper_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"

    # ── Ses ────────────────────────────────────────────────────
    sample_rate: int = 16000
    chunk_duration_s: float = 1  # Düşük gecikme için 1s (eskiden 1.5)
    max_buffer_s: float = 8.0  # Kayan pencere sınırı (8s → maks ~800ms işlem süresi)
    energy_threshold: float = 0.005  # RMS enerji kapısı
    end_of_speech_s: float = 1.2  # Konuşma sonu sessizlik eşiği (saniye) - Nefes alma/düşünme payı için 1.2s idealdir.
    max_speech_s: float = 8.0  # Kesintisiz konuşma sınırı → zorla transkript et
    force_transcription_on_max_speech: bool = (
        False  # Zorla transkript etmeyi açıp kapatmak için
    )

    # ── Whisper Ayarları ───────────────────────────────────────
    beam_size: int = 16  # Maksimum hız için 1 (Greedy decoding). 5 çok yavaştır.
    temperature: float = 0.0  # Deterministik çıktı
    language: str = "tr"
    vad_filter: bool = True
    vad_min_silence_ms: int = 600  # VAD sessizlik eşiği
    word_timestamps: bool = True  # Hizalama zorunluluğu → halüsinasyon azaltır
    suppress_blank: bool = True  # Boş token bastırma
    condition_on_previous_text: bool = False  # Geri besleme döngüsünü kır

    # Bankacılık alanı ön-promptu
    initial_prompt: str = BANKING_INITIAL_PROMPT

    # ── Güven Eşikleri ─────────────────────────────────────────
    no_speech_threshold: float = 0.3  # Sıkı (eskiden 0.5)
    avg_logprob_threshold: float = -0.7  # Bankacılık için sıkı
    compression_ratio_max: float = 2.0  # Tekrarlı döngüleri yakala
    accept_confidence: float = 0.80  # ≥ 0.75 → ACCEPTED
    low_confidence_min: float = 0.50  # 0.50-0.74 → LOW_CONFIDENCE, < 0.50 → REJECTED

    # ── Ollama Dolgu Kelimesi (Filler) Temizleme ───────────────
    use_ollama: bool = False
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    ollama_model: str = "qwen3.5:0.8b"
    silence_reset_s: float = 15.0  # Otomatik sıfırlama süresi (kullanım dışı)
