"""
Voice Consistency Monitor

Tracks vocal identity consistency across model updates and
detects drift from the reference identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from aether.voice.identity.blueprint import VocalIdentity


@dataclass
class DriftReport:
    """Report from a drift check."""
    status: str  # "PASSED", "WARNING", "BLOCKED"
    cumulative_drift: float
    version_drift: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConsistencyReport:
    """Report from a consistency check."""
    identity_drift: float
    max_allowed_drift: float
    is_consistent: bool
    per_dimension_drift: List[float]
    recommendations: List[str] = field(default_factory=list)


class VoiceConsistencyMonitor:
    """
    Monitors for identity drift across model updates.

    Compares synthesized outputs against reference samples to ensure
    the voice maintains its characteristic identity.
    """

    def __init__(
        self,
        reference_identity: VocalIdentity,
        reference_embeddings: Optional[List[np.ndarray]] = None,
        max_drift_per_version: float = 0.05,
        max_cumulative_drift: float = 0.15,
    ):
        """
        Initialize the consistency monitor.

        Args:
            reference_identity: The canonical vocal identity
            reference_embeddings: Optional pre-computed embeddings of reference samples
            max_drift_per_version: Maximum allowed drift per model version
            max_cumulative_drift: Maximum total drift from v1.0
        """
        self.reference_identity = reference_identity
        self.reference_vector = reference_identity.get_identity_vector()

        if reference_embeddings is not None:
            self.reference_embeddings = reference_embeddings
            self.reference_centroid = np.mean(reference_embeddings, axis=0)
        else:
            self.reference_embeddings = None
            self.reference_centroid = None

        self.max_drift_per_version = max_drift_per_version
        self.max_cumulative_drift = max_cumulative_drift

    def check_identity_consistency(
        self,
        test_identity: VocalIdentity
    ) -> ConsistencyReport:
        """
        Check if a test identity is consistent with the reference.

        Args:
            test_identity: The identity to test

        Returns:
            ConsistencyReport with drift details
        """
        test_vector = test_identity.get_identity_vector()

        # Calculate per-dimension drift
        per_dim_drift = np.abs(test_vector - self.reference_vector)

        # Overall drift (L2 norm, normalized)
        total_drift = np.linalg.norm(per_dim_drift) / np.sqrt(len(per_dim_drift))

        is_consistent = total_drift <= self.max_drift_per_version

        recommendations = []
        if not is_consistent:
            # Find dimensions with highest drift
            high_drift_dims = np.argsort(per_dim_drift)[-3:]
            dim_names = self._get_dimension_names()
            for dim_idx in high_drift_dims:
                if per_dim_drift[dim_idx] > 0.05:
                    recommendations.append(
                        f"High drift in {dim_names[dim_idx]}: {per_dim_drift[dim_idx]:.3f}"
                    )

        return ConsistencyReport(
            identity_drift=total_drift,
            max_allowed_drift=self.max_drift_per_version,
            is_consistent=is_consistent,
            per_dimension_drift=per_dim_drift.tolist(),
            recommendations=recommendations,
        )

    def check_embedding_consistency(
        self,
        test_embeddings: List[np.ndarray]
    ) -> ConsistencyReport:
        """
        Check consistency using audio embeddings.

        Args:
            test_embeddings: Embeddings from test audio samples

        Returns:
            ConsistencyReport with drift details
        """
        if self.reference_centroid is None:
            raise ValueError("No reference embeddings available")

        test_centroid = np.mean(test_embeddings, axis=0)

        # Calculate drift from reference centroid
        drift = np.linalg.norm(test_centroid - self.reference_centroid)

        # Per-sample drift
        per_sample_drift = [
            np.linalg.norm(emb - self.reference_centroid)
            for emb in test_embeddings
        ]

        is_consistent = drift <= self.max_drift_per_version

        recommendations = []
        if not is_consistent:
            recommendations.append(
                f"Centroid drift ({drift:.4f}) exceeds threshold ({self.max_drift_per_version})"
            )
            if max(per_sample_drift) > drift * 1.5:
                recommendations.append(
                    "High variance in per-sample drift - consider more consistent training"
                )

        return ConsistencyReport(
            identity_drift=drift,
            max_allowed_drift=self.max_drift_per_version,
            is_consistent=is_consistent,
            per_dimension_drift=per_sample_drift,
            recommendations=recommendations,
        )

    def _get_dimension_names(self) -> List[str]:
        """Get names for identity vector dimensions."""
        return [
            "tessitura_low", "tessitura_high",
            "f1", "f2", "f3", "singers_formant",
            "brightness", "breathiness", "grit", "nasality",
            "chest_resonance", "head_voice_blend",
            "warmth", "control", "intimacy", "power_reserve",
            "sincerity", "engagement",
            "vibrato_rate", "vibrato_onset",
            "transition_smoothness", "consonant_clarity", "sibilance_brightness",
            "reserved",
        ]


class IdentityDriftTracker:
    """
    Tracks identity drift across multiple versions.

    Maintains history of drift measurements to detect gradual
    drift that might not trigger per-version alerts.
    """

    def __init__(
        self,
        reference_identity: VocalIdentity,
        max_cumulative_drift: float = 0.15,
    ):
        """
        Initialize the drift tracker.

        Args:
            reference_identity: The v1.0 reference identity
            max_cumulative_drift: Maximum total drift budget
        """
        self.reference_identity = reference_identity
        self.reference_vector = reference_identity.get_identity_vector()
        self.max_cumulative_drift = max_cumulative_drift
        self.drift_history: List[DriftReport] = []

    def check_version(
        self,
        new_identity: VocalIdentity,
        version: str = "unknown"
    ) -> DriftReport:
        """
        Check a new model version for drift.

        Args:
            new_identity: The identity from the new model version
            version: Version identifier for tracking

        Returns:
            DriftReport with pass/fail status
        """
        new_vector = new_identity.get_identity_vector()

        # Calculate cumulative drift from v1.0
        cumulative_drift = np.linalg.norm(new_vector - self.reference_vector)

        # Calculate version-to-version drift
        if self.drift_history:
            last_vector = self.drift_history[-1].details.get("vector")
            if last_vector is not None:
                version_drift = np.linalg.norm(new_vector - np.array(last_vector))
            else:
                version_drift = cumulative_drift
        else:
            version_drift = cumulative_drift

        # Determine status
        if cumulative_drift > self.max_cumulative_drift:
            status = "BLOCKED"
            message = f"Cumulative identity drift ({cumulative_drift:.4f}) exceeds budget ({self.max_cumulative_drift})"
        elif cumulative_drift > self.max_cumulative_drift * 0.8:
            status = "WARNING"
            message = f"Approaching drift budget: {cumulative_drift:.4f} / {self.max_cumulative_drift}"
        else:
            status = "PASSED"
            message = f"Identity preserved. Drift: {cumulative_drift:.4f}"

        report = DriftReport(
            status=status,
            cumulative_drift=cumulative_drift,
            version_drift=version_drift,
            message=message,
            details={
                "version": version,
                "vector": new_vector.tolist(),
            },
        )

        self.drift_history.append(report)
        return report

    def get_drift_trend(self) -> Dict[str, float]:
        """
        Analyze drift trend over time.

        Returns:
            Dictionary with trend statistics
        """
        if len(self.drift_history) < 2:
            return {"trend": 0.0, "avg_per_version": 0.0}

        cumulative_drifts = [r.cumulative_drift for r in self.drift_history]
        version_drifts = [r.version_drift for r in self.drift_history]

        # Calculate trend (positive = increasing drift)
        if len(cumulative_drifts) >= 3:
            recent_avg = np.mean(cumulative_drifts[-3:])
            early_avg = np.mean(cumulative_drifts[:3])
            trend = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0
        else:
            trend = 0.0

        return {
            "trend": trend,
            "avg_per_version": np.mean(version_drifts),
            "max_drift": max(cumulative_drifts),
            "current_drift": cumulative_drifts[-1] if cumulative_drifts else 0.0,
            "budget_used": cumulative_drifts[-1] / self.max_cumulative_drift if cumulative_drifts else 0.0,
        }

    def should_retrain_from_reference(self) -> bool:
        """
        Determine if the model should be retrained from the original reference.

        Returns:
            True if drift is too high and retraining is recommended
        """
        if not self.drift_history:
            return False

        current_drift = self.drift_history[-1].cumulative_drift
        return current_drift > self.max_cumulative_drift * 0.9
