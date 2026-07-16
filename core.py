"""
Bankacılık düzeyinde Türkçe Konuşmadan Metne (STT) pipeline'ı.

Mimari:
  STTPipeline (sınıf)  → İç OOP motor
  run_stt_pipeline()   → Geriye uyumlu fonksiyon arayüzü (main.py bunu çağırır)

Felsefe: TAHMİN ETME, REDDET.
  Güven düşükse → "Anlayamadım, tekrar eder misiniz?" demek,
  yanlış transkripsiyon vermekten her zaman daha iyidir.
"""

import os
import time
import queue
import logging
from typing import Callable, Optional

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

from config import STTConfig, BANKING_INITIAL_PROMPT
from models import TranscriptionResult, TranscriptionStatus, SegmentDetail
from filters import HallucinationFilter

# ── Ortam Değişkenleri ─────────────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# STTPipeline — Ana Motor
# ═══════════════════════════════════════════════════════════════════════════


class STTPipeline:
    """
    Bankacılık düzeyinde gerçek zamanlı STT pipeline'ı.

    Özellikler:
      - Model ısınması (JIT derleme için sessiz ses transkripsiyon)
      - Kayan pencere tampon (sonsuz büyümeyi engeller)
      - Çok katmanlı halüsinasyon filtresi
      - Enerji kapısı (sessizliği Whisper'a göndermez)
      - Güven bazlı sonuç sınıflandırma (ACCEPTED/LOW_CONFIDENCE/REJECTED/SILENCE)
      - Kuyruk taşma koruması
    """

    def __init__(
        self,
        config: STTConfig,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
    ):
        self.config = config
        self.on_result = on_result or self._default_callback
        self.halucination_filter = HallucinationFilter(config)

        # Ses durumu
        self._buffer = np.array([], dtype=np.float32)
        self._audio_queue: queue.Queue = queue.Queue(maxsize=200)
        self._max_buffer_samples = int(config.sample_rate * config.max_buffer_s)

        # Konuşma tespiti durumu
        self._speech_active: bool = False
        self._silence_samples: int = 0
        self._end_of_speech_samples = int(config.sample_rate * config.end_of_speech_s)
        self._max_speech_samples = int(config.sample_rate * config.max_speech_s)

        # Model indirme durumunu kontrol et
        local_exists = self._check_local_model_exists(config.whisper_model)
        if local_exists:
            logger.info(
                "Model yerel diskte bulundu. İnternet kontrolü bypass ediliyor (local_files_only=True)."
            )

        # Model yükle
        logger.info(
            "Model yükleniyor: %s (%s/%s)",
            config.whisper_model,
            config.device,
            config.compute_type,
        )
        self._model = WhisperModel(
            config.whisper_model,
            device=config.device,
            compute_type=config.compute_type,
            local_files_only=local_exists,
        )

        # CUDA JIT derlemesini tetikle
        self._warmup()
        logger.info("Pipeline hazır.")

    # ── Model Kontrolü  ─────────────────────────────────────────────────────
    def _check_local_model_exists(self, model_name: str) -> bool:
        """
        Hugging Face cache dizinini kontrol ederek modelin yerelde olup olmadığını denetler.
        """
        # Kullanıcının model indirdiği varsayılan dizin
        cache_dir = os.path.expanduser(r"~/.cache/huggingface/hub")

        # Kısa model isimlerinin repo karşılıkları
        repo_map = {
            "tiny": "Systran/faster-whisper-tiny",
            "tiny.en": "Systran/faster-whisper-tiny.en",
            "base": "Systran/faster-whisper-base",
            "base.en": "Systran/faster-whisper-base.en",
            "small": "Systran/faster-whisper-small",
            "small.en": "Systran/faster-whisper-small.en",
            "medium": "Systran/faster-whisper-medium",
            "medium.en": "Systran/faster-whisper-medium.en",
            "large-v1": "Systran/faster-whisper-large-v1",
            "large-v2": "Systran/faster-whisper-large-v2",
            "large-v3": "Systran/faster-whisper-large-v3",
            "large": "Systran/faster-whisper-large-v3",
        }

        repo_id = repo_map.get(model_name, model_name)
        formatted_repo = f"models--{repo_id.replace('/', '--')}"
        model_path = os.path.join(cache_dir, formatted_repo)

        # Dizin ve snapshot kontrolü
        if os.path.exists(model_path):
            snapshots_dir = os.path.join(model_path, "snapshots")
            if os.path.exists(snapshots_dir) and os.listdir(snapshots_dir):
                return True

        # Doğrudan yerel bir klasör yolu verilmişse
        if os.path.exists(model_name):
            return True

        return False

    # ── Model Isınması ─────────────────────────────────────────────────────

    def _warmup(self):
        """1 saniyelik sessiz ses ile CUDA JIT derlemesini tetikle.
        İlk gerçek transkripsiyon böylece hızlı olur."""
        logger.info("Model ısınma çalıştırılıyor...")
        warmup_audio = np.zeros(self.config.sample_rate, dtype=np.float32)
        try:
            segments, _ = self._model.transcribe(
                warmup_audio,
                language=self.config.language,
                beam_size=1,  # Hızlı ısınma
                vad_filter=False,  # VAD'a gerek yok
            )
            # Jeneratörü tüket → gerçekten çıkarım çalışsın
            for _ in segments:
                pass
            logger.info("Model ısınması tamamlandı.")
        except Exception as e:
            logger.warning("Isınma hatası (göz ardı ediliyor): %s", e)

    # ── Ses Giriş Yönetimi ────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — veriyi kuyruğa ekle, engelleme yapma."""
        if status:
            logger.warning("Mikrofon hatası: %s", status)
        try:
            self._audio_queue.put_nowait(indata.copy())
        except queue.Full:
            # Kuyruk doluysa en eski chunk'ı at, yenisini ekle
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(indata.copy())
            except queue.Full:
                logger.warning("Audio kuyruk taşması — chunk atlandı")

    def _drain_queue(self) -> np.ndarray:
        """Kuyruktan tüm bekleyen ses verilerini oku. Gecikme birikimini önler."""
        chunks = []
        try:
            while True:
                chunks.append(np.squeeze(self._audio_queue.get_nowait()))
        except queue.Empty:
            pass

        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.float32)

    # ── Tampon Yönetimi ────────────────────────────────────────────────────

    def _trim_buffer(self):
        """Kayan pencere: tamponu max_buffer_samples'a kırp."""
        if len(self._buffer) > self._max_buffer_samples:
            excess = len(self._buffer) - self._max_buffer_samples
            self._buffer = self._buffer[excess:]
            logger.debug("Tampon kırpıldı: %d örnek silindi", excess)

    def _clear_buffer(self):
        """Tampon ve konuşma durumunu sıfırla."""
        self._buffer = np.array([], dtype=np.float32)
        self._speech_active = False
        self._silence_samples = 0
        self.halucination_filter.reset()

    # ── Transkripsiyon ─────────────────────────────────────────────────────

    def _transcribe(self) -> TranscriptionResult:
        """Tam filtreleme zinciri ile transkripsiyon çalıştır."""
        start_time = time.perf_counter()

        segments, _info = self._model.transcribe(
            self._buffer,
            language=self.config.language,
            beam_size=self.config.beam_size,
            temperature=self.config.temperature,
            vad_filter=self.config.vad_filter,
            vad_parameters=(
                dict(min_silence_duration_ms=self.config.vad_min_silence_ms)
                if self.config.vad_filter
                else None
            ),
            initial_prompt=(
                self.config.initial_prompt if self.config.initial_prompt else None
            ),
            condition_on_previous_text=self.config.condition_on_previous_text,
            word_timestamps=self.config.word_timestamps,
            suppress_blank=self.config.suppress_blank,
        )

        # Segmentleri filtrele
        valid_segments, confidence = self.halucination_filter.filter_segments(segments)

        # Metni birleştir
        text = " ".join(seg.text for seg in valid_segments).strip()

        processing_ms = (time.perf_counter() - start_time) * 1000.0

        return self.halucination_filter.build_result(
            text,
            valid_segments,
            confidence,
            processing_ms,
        )

    # ── Dolgu Kelimesi Temizleme (Ollama) ──────────────────────────────────

    def _clean_filler_words(self, text: str) -> str:
        """Ollama ile metinden 'eee, hmm' gibi dolgu kelimelerini temizle."""
        prompt = (
            "Görevin: Aşağıdaki metinden 'eee', 'ııı', 'hmm', 'şey' gibi düşünme belirten "
            "anlamsız sesleri ve dolgu (filler) kelimelerini temizlemektir.\n"
            "Cümlenin asıl anlamını bozma, başka kelime ekleme veya çıkarma.\n"
            "Sadece temizlenmiş metni döndür, başka hiçbir açıklama veya yorum yapma.\n"
            f'Orijinal Metin: "{text}"\n'
            "Temizlenmiş Metin:"
        )
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            start_t = time.perf_counter()
            resp = requests.post(
                self.config.ollama_url,
                json=payload,
                timeout=5,
            ).json()
            result = resp.get("response", "").strip()
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.debug("Ollama yanıt: '%s' (%.0fms)", result, elapsed_ms)

            return result
        except Exception as e:
            logger.error("Ollama hatası: %s", e)
            return text

    # ── Varsayılan Callback ────────────────────────────────────────────────

    @staticmethod
    def _default_callback(result: TranscriptionResult):
        """Renkli terminal çıktısı ile varsayılan sonuç işleyici."""
        if result.status == TranscriptionStatus.SILENCE:
            return

        status_map = {
            TranscriptionStatus.ACCEPTED: "\n✅ [{conf:.0%} | {ms:.0f}ms] {text}",
            TranscriptionStatus.LOW_CONFIDENCE: "\n⚠️  [{conf:.0%} | {ms:.0f}ms] {text}  (doğrulama gerekli)",
            TranscriptionStatus.REJECTED: "",  # Reddedilenleri kullanıcıya gösterme
        }

        template = status_map.get(result.status, "")
        if template:
            print(
                template.format(
                    conf=result.confidence,
                    ms=result.processing_time_ms,
                    text=result.text,
                )
            )
        elif result.status == TranscriptionStatus.REJECTED:
            logger.info(
                "Reddedildi [%.0f%%]: '%s'",
                result.confidence * 100,
                result.text,
            )

    def _do_transcribe_and_output(self):
        """
        Konuşma sonu tespit edildiğinde çağrılır.
        Tamponu TEK SEFERDE transkript eder, sonucu iletir,
        gerekirse Ollama ile dolgu kelimelerini temizler ve tamponu temizler.
        """
        if self._buffer.size == 0:
            return

        result = self._transcribe()

        if result.status == TranscriptionStatus.SILENCE or not result.text:
            # Ses vardı ama anlamlı metin çıkmadı → temizle, devam et
            self._clear_buffer()
            return

        # Sonucu ilet
        self.on_result(result)

        full_text = result.text.strip()

        # ── Ollama Dolgu Kelimesi Temizleme ───────────────────────────
        if self.config.use_ollama:
            cleaned_text = self._clean_filler_words(full_text)
            print(f"\n[Ollama ile Temizlenmiş Metin]: {cleaned_text}")
            full_text = cleaned_text

        print(f"\n>>> NİHAİ İSTEK YAKALANDI: {full_text}\n")
        print("--- Yeni Cümle Bekleniyor ---\n")

        # Ses tamponunu her zaman temizle → bir sonraki konuşma taze başlar
        self._clear_buffer()

    # ── Ana Döngü ──────────────────────────────────────────────────────────

    def start(self):
        """
        Pipeline'ı başlat. KeyboardInterrupt'a kadar engeller.

        Mimari: KONUŞMA-BAZLI İŞLEME
          1. Ses varsa → biriktir (transkripsiyon YAPMA)
          2. Sessizlik 0.8s aştı → konuşma bitti → TEK SEFERDE transkript et
          3. Tamponu temizle → sonraki konuşmayı bekle

        Bu sayede her cümle sadece 1 kez işlenir. Sıfır tekrar.
        """
        print("Sistem Hazır! Konuşmaya başlayın (Çıkmak için CTRL+C)\n")

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        ):
            try:
                while True:
                    # İlk chunk'ı bekle (engelleyici)
                    first_chunk = np.squeeze(self._audio_queue.get())

                    # Kuyrukta biriken ek chunk'ları hemen oku
                    extra = self._drain_queue()
                    new_audio = (
                        np.concatenate([first_chunk, extra])
                        if extra.size > 0
                        else first_chunk
                    )

                    has_energy = self.halucination_filter.check_energy(new_audio)

                    # Her durumda tampona ekle
                    self._buffer = np.concatenate((self._buffer, new_audio))
                    self._trim_buffer()

                    if has_energy:
                        # ── Konuşma algılandı ────────────────────────
                        if not self._speech_active:
                            self._speech_active = True
                            print("🎙️", end="", flush=True)
                        self._silence_samples = 0

                        # Güvenlik: çok uzun kesintisiz konuşma → zorla transkript et
                        if self.config.force_transcription_on_max_speech and len(self._buffer) >= self._max_speech_samples:
                            logger.info(
                                "Maks konuşma süresi aşıldı (%.1fs), zorla transkript ediliyor.",
                                self.config.max_speech_s,
                            )
                            self._do_transcribe_and_output()

                    elif self._speech_active:
                        # ── Konuşuyordu, şimdi sessiz ─────────────────
                        self._silence_samples += len(new_audio)

                        if self._silence_samples >= self._end_of_speech_samples:
                            # Konuşma sonu! Tek seferde transkript et.
                            self._do_transcribe_and_output()

                    # else: konuşma yok, enerji yok → beklemeye devam
            except KeyboardInterrupt:
                print("\nÇıkış yapılıyor...")
    def stop(self):
        """Pipeline'ı durdur ve kaynakları temizle."""
        self._clear_buffer()
        logger.info("Pipeline durduruldu.")


# ═══════════════════════════════════════════════════════════════════════════
# clean_filler_words — Geriye Uyumlu Fonksiyon
# ═══════════════════════════════════════════════════════════════════════════


def clean_filler_words(text, ollama_url, ollama_model):
    """Eski arayüz yerine dolgu kelimesi temizleme için güncellendi."""
    prompt = (
        "Görevin: Aşağıdaki metinden 'eee', 'ııı', 'hmm', 'şey' gibi düşünme belirten "
        "anlamsız sesleri ve dolgu (filler) kelimelerini temizlemektir.\n"
        "Cümlenin asıl anlamını bozma, başka kelime ekleme veya çıkarma.\n"
        "Sadece temizlenmiş metni döndür, başka hiçbir açıklama veya yorum yapma.\n"
        f'Orijinal Metin: "{text}"\n'
        "Temizlenmiş Metin:"
    )
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        start_t = time.time()
        response = requests.post(ollama_url, json=payload, timeout=5).json()
        result = response.get("response", "").strip()
        print(f" [Ollama Temizleme Hızı: {time.time() - start_t:.2f}s]")
        return result
    except Exception as e:
        print(f"[Ollama Hatası]: {e}")
        return text


# ═══════════════════════════════════════════════════════════════════════════
# run_stt_pipeline — Geriye Uyumlu Ana Arayüz
# ═══════════════════════════════════════════════════════════════════════════


def run_stt_pipeline(
    whisper_model="large-v3",
    device="cpu",
    compute_type="int8",
    use_ollama=True,
    ollama_url="http://localhost:11434/api/generate",
    ollama_model="qwen3.5:0.8b",
    rate=16000,
    chunk_duration_s=1.5,
    silence_threshold=0.01,
    initial_prompt="",
    language="tr",
    vad_filter=True,
    word_timestamps=True,
    # ── Yeni bankacılık düzeyi parametreler (varsayılanlarla) ──────────
    beam_size=5,
    max_buffer_s=15.0,
    energy_threshold=None,
    no_speech_threshold=0.3,
    avg_logprob_threshold=-0.7,
    compression_ratio_max=2.0,
    accept_confidence=0.75,
    low_confidence_min=0.50,
    vad_min_silence_ms=300,
    force_transcription_on_max_speech=True,
    on_result=None,
):
    """
    Bankacılık düzeyinde Türkçe STT pipeline'ı.

    Geriye uyumlu fonksiyon arayüzü — main.py'deki çağrı şekli korunur.
    Tüm orijinal parametreler aynen kabul edilir. Yeni parametreler
    halüsinasyon filtreleme sistemini kontrol eder.

    Args:
        whisper_model: Whisper model adı (large-v3, distil-large-v3, vb.)
        device: Hesaplama cihazı (cuda, cpu)
        compute_type: Hesaplama tipi (float16, int8, vb.)
        use_ollama: Ollama dolgu kelimesi temizlemeyi aç/kapat (True/False)
        ollama_url: Ollama API adresi
        ollama_model: Ollama model adı
        rate: Ses örnekleme hızı (Hz)
        chunk_duration_s: Anlık işleme süresi (saniye)
        silence_threshold: Sessizlik eşiği (enerji kapısı olarak kullanılır)
        initial_prompt: Whisper ön-promptu (boşsa bankacılık promptu kullanılır)
        language: Dil kodu
        vad_filter: VAD filtresi aç/kapat
        word_timestamps: Kelime zaman damgaları

        beam_size: Beam genişliği (doğruluk için 5+)
        max_buffer_s: Maksimum tampon süresi (saniye)
        energy_threshold: RMS enerji kapısı eşiği (None ise silence_threshold kullanılır)
        no_speech_threshold: no_speech_prob üst sınırı
        avg_logprob_threshold: avg_logprob alt sınırı
        compression_ratio_max: compression_ratio üst sınırı
        accept_confidence: ACCEPTED durum güven eşiği
        low_confidence_min: LOW_CONFIDENCE alt sınırı
        vad_min_silence_ms: VAD minimum sessizlik süresi (ms)
        on_result: Sonuç callback'i (None ise varsayılan terminal çıktısı)
    """
    # energy_threshold belirtilmemişse, silence_threshold'u kullan
    effective_energy = (
        energy_threshold if energy_threshold is not None else silence_threshold
    )

    # Boş initial_prompt → bankacılık alanı promptu
    effective_prompt = initial_prompt if initial_prompt else BANKING_INITIAL_PROMPT

    config = STTConfig(
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        sample_rate=rate,
        chunk_duration_s=chunk_duration_s,
        max_buffer_s=max_buffer_s,
        energy_threshold=effective_energy,
        beam_size=beam_size,
        temperature=0.0,
        language=language,
        vad_filter=vad_filter,
        vad_min_silence_ms=vad_min_silence_ms,
        word_timestamps=word_timestamps,
        suppress_blank=True,
        condition_on_previous_text=False,
        initial_prompt=effective_prompt,
        no_speech_threshold=no_speech_threshold,
        avg_logprob_threshold=avg_logprob_threshold,
        compression_ratio_max=compression_ratio_max,
        accept_confidence=accept_confidence,
        low_confidence_min=low_confidence_min,
        use_ollama=use_ollama,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        silence_reset_s=15.0,
        force_transcription_on_max_speech=force_transcription_on_max_speech,
    )

    # Logging ayarla (INFO seviyesi — DEBUG çok gürültülü)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # faster_whisper'ın DEBUG spam'ini sustur
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    print(f"[{whisper_model}] Yükleniyor...")
    pipeline = STTPipeline(config=config, on_result=on_result)
    pipeline.start()
