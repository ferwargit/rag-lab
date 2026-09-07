"""Compatibilidad temporal: la evaluación de evidencia vive en evaluation."""

from rag_lab.evaluation.evidence import (
    EvidenceAssessment,
    assess_evidence,
)

__all__ = [
    "EvidenceAssessment",
    "assess_evidence",
]