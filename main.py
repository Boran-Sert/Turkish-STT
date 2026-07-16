from core import run_stt_pipeline

if __name__ == "__main__":
    # ── Test ortamı parametlerini değiştirebilirsiniz (Detaylı ayaları config.py dosyasından incleyebilirsiniz) ──────────────────────────────────────────
    run_stt_pipeline(
        # ── 1. Model Yapılandırması ──────────────────────────────────────
        whisper_model="large-v3",
        device="cuda",                      # "cuda" veya "cpu"
        compute_type="float16",             # float16, int8, bfloat16, int4

        # ── 2. Ses Akışı Yönetimi ────────────────────────────────────────
        rate=16000,                         # Örnekleme hızı (Hz)
        chunk_duration_s=1.5,               # Anlık işleme paket süresi (saniye)
        silence_threshold=0.01,             # RMS gürültü/sessizlik kapısı
        force_transcription_on_max_speech=False, # Maksimum süre aşımında zorla çeviri

        # ── 3. Whisper Çıkarım Ayarları ──────────────────────────────────
        initial_prompt="",                  # Kelime dağarcığı yönlendirme (Boşsa varsayılan bankacılık promptu)
        language="tr",                      # Hedef dil
        beam_size=16,                       # Arama genişliği (Yüksekse hassas ama yavaş, 1 veya 2 hızlıdır)

        # ── 4. Ses Aktivite Filtresi (VAD) ───────────────────────────────
        vad_filter=True,                    # VAD filtresini aktif et
        vad_min_silence_ms=200,             # VAD minimum sessizlik süresi (ms)
        word_timestamps=True,               # Kelime zaman damgaları (Zorunlu - halüsinasyon engeller)

        # ── 5. Ollama Dolgu Kelimesi Temizleme ───────────────────────────
        use_ollama=False,                   # Ollama ile 'eee, hmm' gibi kelimeleri temizleme
        ollama_url="http://localhost:11434/api/generate",
        ollama_model="qwen3.5:0.8b",
    )
