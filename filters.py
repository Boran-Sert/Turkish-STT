"""
Çok katmanlı halüsinasyon filtreleme motoru.

Katman sırası:
  1. RMS Enerji Kapısı     → Sessiz ses Whisper'a gönderilmez
  2. Whisper VAD            → Silero VAD (transcribe içinde)
  3. no_speech_prob         → Yüksekse segment reddedilir
  4. avg_logprob            → Düşükse segment reddedilir
  5. compression_ratio      → Yüksekse tekrarlı halüsinasyon
  6. Kara Liste             → Türkçe halüsinasyon kalıpları
  7. Tekrar Tespiti         → Aynı metin art arda → reddedilir
  8. Minimum Uzunluk        → Çok kısa segment → reddedilir
  9. Güven Birleştirme      → Ağırlıklı skor → durum kararı
"""

import logging
from typing import List, Tuple

import numpy as np

from config import TURKISH_HALLUCINATION_PATTERNS, STTConfig
from models import SegmentDetail, TranscriptionResult, TranscriptionStatus

logger = logging.getLogger(__name__)


class HallucinationFilter:
    """Bankacılık düzeyinde çok katmanlı filtreleme."""

    def __init__(self, config: STTConfig):
        self.config = config
        self._last_texts: List[str] = []
        self._max_history: int = 5

    # ── Katman 1: Enerji Kapısı ────────────────────────────────────────────

    def check_energy(self, audio: np.ndarray) -> bool:
        """
        RMS enerji kontrolü. Sessiz ses için False döner → transkripsiyon atlanır.
        Bu tek başına halüsinasyonların %60-70'ini engeller.
        """
        if audio.size == 0:
            return False
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.config.energy_threshold:
            logger.debug("Enerji kapısı: RMS %.6f < %.4f → ATLA", rms, self.config.energy_threshold)
            return False
        return True

    # ── Katman 6: Kara Liste ───────────────────────────────────────────────

    def is_hallucination(self, text: str) -> bool:
        """Türkçe halüsinasyon kara listesine karşı kontrol."""
        text_stripped = text.strip()
        if not text_stripped:
            return True

        for pattern in TURKISH_HALLUCINATION_PATTERNS:
            if pattern.search(text_stripped):
                logger.info("Halüsinasyon yakalandı: '%s' → kalıp '%s'", text_stripped, pattern.pattern)
                return True
        return False

    # ── Katman 7: Tekrar Tespiti ───────────────────────────────────────────

    def _detect_repetition(self, text: str) -> bool:
        """Art arda aynı metin 3+ kez tekrarlanıyorsa → döngü halüsinasyonu."""
        text_clean = text.strip().lower()

        if len(self._last_texts) >= 3:
            if all(t == text_clean for t in self._last_texts[-3:]):
                logger.info("Tekrar döngüsü tespit edildi: '%s'", text)
                return True

        self._last_texts.append(text_clean)
        if len(self._last_texts) > self._max_history:
            self._last_texts.pop(0)

        return False

    # ── Segment Güven Skoru ────────────────────────────────────────────────

    @staticmethod
    def _segment_confidence(no_speech_prob: float, avg_logprob: float,
                            compression_ratio: float) -> float:
        """
        Tek segment için 0.0–1.0 arası güven skoru hesapla.

        Ağırlıklar:
          - avg_logprob   → %50  (en güvenilir sinyal)
          - no_speech_prob → %30  (konuşma var mı?)
          - compression    → %20  (tekrar var mı?)
        """
        # avg_logprob: -0.7 → ~0.3, 0.0 → 1.0
        logprob_score = max(0.0, min(1.0, (avg_logprob + 1.0) / 1.0))

        # no_speech_prob: 0.0 → 1.0, 0.3 → 0.7
        speech_score = 1.0 - no_speech_prob

        # compression_ratio: 1.0 → 1.0, 2.0 → 0.5, 4.0 → 0.25
        compression_score = max(0.0, min(1.0, 1.0 / max(compression_ratio, 0.01)))

        return logprob_score * 0.5 + speech_score * 0.3 + compression_score * 0.2

    # ── Ana Filtreleme ─────────────────────────────────────────────────────

    def filter_segments(self, segments) -> Tuple[List[SegmentDetail], float]:
        """
        Whisper segmentlerini çok katmanlı filtreden geçir.

        Döner:
            (geçerli_segmentler, birleşik_güven_skoru)
        """
        valid: List[SegmentDetail] = []
        confidence_scores: List[float] = []

        for seg in segments:
            # Katman 3: no_speech_prob
            if seg.no_speech_prob > self.config.no_speech_threshold:
                logger.debug(
                    "Segment reddedildi: no_speech_prob %.3f > %.2f",
                    seg.no_speech_prob, self.config.no_speech_threshold,
                )
                continue

            # Katman 4: avg_logprob
            if seg.avg_logprob < self.config.avg_logprob_threshold:
                logger.debug(
                    "Segment reddedildi: avg_logprob %.3f < %.2f",
                    seg.avg_logprob, self.config.avg_logprob_threshold,
                )
                continue

            # Katman 5: compression_ratio
            if seg.compression_ratio > self.config.compression_ratio_max:
                logger.debug(
                    "Segment reddedildi: compression_ratio %.2f > %.1f",
                    seg.compression_ratio, self.config.compression_ratio_max,
                )
                continue

            # Katman 6: Kara liste
            if self.is_hallucination(seg.text):
                continue

            # Katman 8: Minimum uzunluk
            if len(seg.text.strip()) < 2:
                logger.debug("Segment reddedildi: çok kısa '%s'", seg.text)
                continue

            # Güven skoru hesapla
            seg_conf = self._segment_confidence(
                seg.no_speech_prob, seg.avg_logprob, seg.compression_ratio,
            )
            confidence_scores.append(seg_conf)

            valid.append(SegmentDetail(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                no_speech_prob=seg.no_speech_prob,
                avg_logprob=seg.avg_logprob,
                compression_ratio=seg.compression_ratio,
            ))

        # Katman 9: Birleşik güven
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )

        return valid, avg_confidence

    # ── Sonuç Oluşturma ───────────────────────────────────────────────────

    def build_result(
        self,
        text: str,
        segments: List[SegmentDetail],
        confidence: float,
        processing_time_ms: float,
    ) -> TranscriptionResult:
        """Filtrelenmiş veriden TranscriptionResult oluştur."""

        if not text or not text.strip():
            return TranscriptionResult(
                text="",
                status=TranscriptionStatus.SILENCE,
                confidence=0.0,
                processing_time_ms=processing_time_ms,
                segments=segments,
            )

        # Katman 7: Tekrar kontrolü
        if self._detect_repetition(text):
            return TranscriptionResult(
                text=text,
                status=TranscriptionStatus.REJECTED,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                segments=segments,
            )

        # Durum kararı
        if confidence >= self.config.accept_confidence:
            status = TranscriptionStatus.ACCEPTED
        elif confidence >= self.config.low_confidence_min:
            status = TranscriptionStatus.LOW_CONFIDENCE
        else:
            status = TranscriptionStatus.REJECTED

        return TranscriptionResult(
            text=text,
            status=status,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            segments=segments,
        )

    # ── Sıfırlama ─────────────────────────────────────────────────────────

    def reset(self):
        """Tampon temizlendiğinde veya uç noktada çağır."""
        self._last_texts.clear()
