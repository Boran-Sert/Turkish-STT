"""
Veri modelleri — TranscriptionResult ve yardımcı tipler.

Her transkripsiyon sonucu bir güven skoru ve durum bilgisi taşır.
Bankacılık uygulaması bu bilgilere göre karar verir:
  ACCEPTED       → Sonucu kullan
  LOW_CONFIDENCE → Kullan ama insan doğrulaması için işaretle
  REJECTED       → Kullanma, müşteriden tekrar etmesini iste
  SILENCE        → Konuşma algılanmadı
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class TranscriptionStatus(Enum):
    """Transkripsiyon sonuç durumu."""
    ACCEPTED = "ACCEPTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    REJECTED = "REJECTED"
    SILENCE = "SILENCE"


@dataclass
class SegmentDetail:
    """Whisper'dan dönen tek bir segment'in detayları."""
    text: str
    start: float
    end: float
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float


@dataclass
class TranscriptionResult:
    """
    Pipeline'ın ürettiği nihai sonuç.

    Bankacılık uygulaması bu nesneyi alır ve status alanına göre
    işlem yapar. confidence alanı denetim izi (audit trail) için
    loglanmalıdır.
    """
    text: str
    status: TranscriptionStatus
    confidence: float                          # 0.0 – 1.0
    processing_time_ms: float                  # Transkripsiyon süresi (ms)
    segments: List[SegmentDetail] = field(default_factory=list)
