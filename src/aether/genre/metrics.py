"""
Genre Differentiation Metrics - Evaluation infrastructure for genre system.

Provides quantitative measurement of genre differentiation quality:
- Embedding separation (cosine distance between genres)
- Rhythm accuracy (pattern matching to DNA)
- Harmonic accuracy (mode/progression matching)
- Overall genre classifier accuracy
"""

from dataclasses import dataclass, field
from typing import Optional
import math

from aether.genre.dna import (
    GenreDNA,
    GENRE_DNA_LIBRARY,
    get_genre_dna,
    compute_genre_similarity,
)
from aether.genre.detection import (
    GenreDetector,
    GenreDetectionResult,
    RhythmFeatures,
    HarmonicFeatures,
)


@dataclass
class GenreMetrics:
    """Metrics for a single genre."""
    genre_id: str

    # Detection accuracy
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Similarity scores
    intra_genre_similarity: float = 0.0  # Similarity within genre
    inter_genre_similarity: float = 0.0  # Similarity to other genres

    # Component scores
    rhythm_accuracy: float = 0.0
    harmony_accuracy: float = 0.0
    melody_accuracy: float = 0.0

    # DNA deviation
    avg_dna_deviation: float = 0.0

    @property
    def precision(self) -> float:
        """Precision score."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Recall score."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """F1 score."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def separation_score(self) -> float:
        """How well-separated is this genre from others."""
        if self.inter_genre_similarity == 0:
            return 1.0
        return self.intra_genre_similarity / self.inter_genre_similarity


@dataclass
class MetricsReport:
    """Complete evaluation report for genre system."""
    # Per-genre metrics
    genre_metrics: dict[str, GenreMetrics] = field(default_factory=dict)

    # Overall metrics
    overall_accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0

    # Embedding metrics
    avg_embedding_separation: float = 0.0
    min_embedding_separation: float = 0.0
    confusion_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    # Component accuracy
    overall_rhythm_accuracy: float = 0.0
    overall_harmony_accuracy: float = 0.0

    # Sample counts
    total_samples: int = 0
    correct_predictions: int = 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "GENRE DIFFERENTIATION METRICS REPORT",
            "=" * 60,
            "",
            f"Total Samples: {self.total_samples}",
            f"Overall Accuracy: {self.overall_accuracy:.2%}",
            f"Macro F1 Score: {self.macro_f1:.2%}",
            "",
            "Embedding Separation:",
            f"  Average: {self.avg_embedding_separation:.3f}",
            f"  Minimum: {self.min_embedding_separation:.3f}",
            "",
            "Component Accuracy:",
            f"  Rhythm: {self.overall_rhythm_accuracy:.2%}",
            f"  Harmony: {self.overall_harmony_accuracy:.2%}",
            "",
            "Per-Genre Performance:",
            "-" * 40,
        ]

        for genre_id, metrics in sorted(self.genre_metrics.items()):
            lines.append(f"  {genre_id}:")
            lines.append(f"    Precision: {metrics.precision:.2%}")
            lines.append(f"    Recall: {metrics.recall:.2%}")
            lines.append(f"    F1: {metrics.f1_score:.2%}")
            lines.append(f"    Separation: {metrics.separation_score:.2f}")

        if self.confusion_pairs:
            lines.append("")
            lines.append("Most Confused Genre Pairs:")
            lines.append("-" * 40)
            for g1, g2, sim in self.confusion_pairs[:5]:
                lines.append(f"  {g1} <-> {g2}: {sim:.3f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


class GenreDifferentiationMetrics:
    """
    Computes and tracks genre differentiation metrics.

    Used to evaluate how well the system distinguishes between genres
    and maintains genre authenticity during generation.
    """

    def __init__(self):
        self.detector = GenreDetector()
        self._samples: list[dict] = []
        self._reset_metrics()

    def _reset_metrics(self):
        """Reset accumulated metrics."""
        self._genre_metrics = {
            genre_id: GenreMetrics(genre_id=genre_id)
            for genre_id in GENRE_DNA_LIBRARY.keys()
        }
        self._confusion_matrix: dict[str, dict[str, int]] = {
            g: {g2: 0 for g2 in GENRE_DNA_LIBRARY.keys()}
            for g in GENRE_DNA_LIBRARY.keys()
        }

    def add_sample(
        self,
        notes: list[dict],
        tempo_bpm: float,
        true_genre: str,
        ticks_per_beat: int = 480,
    ):
        """
        Add a sample for evaluation.

        Args:
            notes: MIDI notes
            tempo_bpm: Track tempo
            true_genre: Ground truth genre label
            ticks_per_beat: MIDI resolution
        """
        # Detect genre
        result = self.detector.detect(
            notes, tempo_bpm, ticks_per_beat, true_genre
        )

        predicted_genre = result.primary_genre

        # Store sample
        self._samples.append({
            "true_genre": true_genre,
            "predicted_genre": predicted_genre,
            "confidence": result.confidence,
            "dna_similarity": result.dna_similarity,
            "rhythm_features": result.rhythm_features,
            "harmonic_features": result.harmonic_features,
        })

        # Update confusion matrix
        if true_genre in self._confusion_matrix:
            if predicted_genre in self._confusion_matrix[true_genre]:
                self._confusion_matrix[true_genre][predicted_genre] += 1

        # Update per-genre metrics
        if true_genre == predicted_genre:
            self._genre_metrics[true_genre].true_positives += 1
        else:
            self._genre_metrics[true_genre].false_negatives += 1
            if predicted_genre in self._genre_metrics:
                self._genre_metrics[predicted_genre].false_positives += 1

    def compute_report(self) -> MetricsReport:
        """Compute full metrics report from accumulated samples."""
        if not self._samples:
            return MetricsReport()

        report = MetricsReport()
        report.total_samples = len(self._samples)
        report.correct_predictions = sum(
            1 for s in self._samples if s["true_genre"] == s["predicted_genre"]
        )
        report.overall_accuracy = report.correct_predictions / report.total_samples

        # Per-genre metrics
        for genre_id, metrics in self._genre_metrics.items():
            # Compute intra/inter genre similarity
            genre_samples = [s for s in self._samples if s["true_genre"] == genre_id]
            other_samples = [s for s in self._samples if s["true_genre"] != genre_id]

            if genre_samples:
                metrics.intra_genre_similarity = sum(
                    s["dna_similarity"] for s in genre_samples
                ) / len(genre_samples)

                # Rhythm accuracy
                metrics.rhythm_accuracy = self._compute_rhythm_accuracy(
                    genre_samples, genre_id
                )

                # Harmony accuracy
                metrics.harmony_accuracy = self._compute_harmony_accuracy(
                    genre_samples, genre_id
                )

            if other_samples:
                # Average similarity of other genres to this one
                similarities = []
                for s in other_samples:
                    result = self.detector.compute_genre_similarity(
                        GenreDetectionResult(
                            primary_genre=s["predicted_genre"],
                            confidence=s["confidence"],
                            genre_scores={},
                            rhythm_features=s["rhythm_features"],
                            harmonic_features=s["harmonic_features"],
                        ),
                        genre_id
                    )
                    similarities.append(result)
                metrics.inter_genre_similarity = sum(similarities) / len(similarities)

            report.genre_metrics[genre_id] = metrics

        # Macro averages
        valid_metrics = [m for m in report.genre_metrics.values()
                        if m.true_positives + m.false_negatives > 0]
        if valid_metrics:
            report.macro_precision = sum(m.precision for m in valid_metrics) / len(valid_metrics)
            report.macro_recall = sum(m.recall for m in valid_metrics) / len(valid_metrics)
            report.macro_f1 = sum(m.f1_score for m in valid_metrics) / len(valid_metrics)
            report.overall_rhythm_accuracy = sum(
                m.rhythm_accuracy for m in valid_metrics
            ) / len(valid_metrics)
            report.overall_harmony_accuracy = sum(
                m.harmony_accuracy for m in valid_metrics
            ) / len(valid_metrics)

        # Embedding separation
        report.avg_embedding_separation, report.min_embedding_separation = \
            compute_embedding_separation()

        # Find most confused pairs
        report.confusion_pairs = self._find_confusion_pairs()

        return report

    def _compute_rhythm_accuracy(
        self,
        samples: list[dict],
        genre_id: str,
    ) -> float:
        """Compute rhythm pattern accuracy for genre samples."""
        dna = get_genre_dna(genre_id)
        if dna is None:
            return 0.0

        scores = []
        for sample in samples:
            rhythm = sample["rhythm_features"]

            # Tempo in range
            tempo_score = 1.0 if (
                dna.rhythm.tempo_range[0] <= rhythm.tempo_bpm <= dna.rhythm.tempo_range[1]
            ) else 0.0

            # Swing match
            swing_diff = abs(rhythm.swing_amount - dna.rhythm.swing_amount)
            swing_score = max(0, 1.0 - swing_diff * 2)

            # Kick pattern match
            kick_score = 1.0 if rhythm.detected_kick_pattern == dna.rhythm.kick_pattern else 0.3

            scores.append((tempo_score + swing_score + kick_score) / 3)

        return sum(scores) / len(scores) if scores else 0.0

    def _compute_harmony_accuracy(
        self,
        samples: list[dict],
        genre_id: str,
    ) -> float:
        """Compute harmonic accuracy for genre samples."""
        dna = get_genre_dna(genre_id)
        if dna is None:
            return 0.0

        scores = []
        for sample in samples:
            harmony = sample["harmonic_features"]

            # Mode match
            if harmony.detected_mode in dna.harmony.primary_modes:
                mode_score = 1.0
            elif harmony.detected_mode in dna.harmony.secondary_modes:
                mode_score = 0.7
            else:
                mode_score = 0.2

            # Complexity in range
            comp = harmony.chord_complexity
            if dna.harmony.chord_complexity[0] <= comp <= dna.harmony.chord_complexity[1]:
                complexity_score = 1.0
            else:
                complexity_score = 0.3

            scores.append((mode_score + complexity_score) / 2)

        return sum(scores) / len(scores) if scores else 0.0

    def _find_confusion_pairs(self) -> list[tuple[str, str, float]]:
        """Find most confused genre pairs from confusion matrix."""
        pairs = []

        for g1 in self._confusion_matrix:
            for g2 in self._confusion_matrix[g1]:
                if g1 < g2:  # Avoid duplicates
                    # Mutual confusion count
                    confusion = (
                        self._confusion_matrix[g1][g2] +
                        self._confusion_matrix[g2][g1]
                    )
                    total_g1 = sum(self._confusion_matrix[g1].values())
                    total_g2 = sum(self._confusion_matrix[g2].values())

                    if total_g1 + total_g2 > 0:
                        confusion_rate = confusion / (total_g1 + total_g2)
                        if confusion_rate > 0:
                            pairs.append((g1, g2, confusion_rate))

        return sorted(pairs, key=lambda x: x[2], reverse=True)

    def reset(self):
        """Reset all accumulated data."""
        self._samples = []
        self._reset_metrics()


def compute_embedding_separation() -> tuple[float, float]:
    """
    Compute embedding separation between all genre pairs.

    Returns:
        (average_separation, minimum_separation)
    """
    genres = list(GENRE_DNA_LIBRARY.keys())
    separations = []

    for i, g1 in enumerate(genres):
        for g2 in genres[i + 1:]:
            # compute_genre_similarity takes genre ID strings, not DNA objects
            similarity = compute_genre_similarity(g1, g2)
            separation = 1.0 - similarity
            separations.append(separation)

    if not separations:
        return 0.0, 0.0

    return sum(separations) / len(separations), min(separations)


def compute_rhythm_accuracy(
    rhythm_features: RhythmFeatures,
    target_genre: str,
) -> float:
    """
    Compute rhythm accuracy for a single detection result.

    Args:
        rhythm_features: Detected rhythm features
        target_genre: Target genre ID

    Returns:
        Accuracy score 0-1
    """
    dna = get_genre_dna(target_genre)
    if dna is None:
        return 0.0

    scores = []

    # Tempo accuracy
    tempo_range = dna.rhythm.tempo_range
    if tempo_range[0] <= rhythm_features.tempo_bpm <= tempo_range[1]:
        mid = (tempo_range[0] + tempo_range[1]) / 2
        range_size = tempo_range[1] - tempo_range[0]
        tempo_score = 1.0 - abs(rhythm_features.tempo_bpm - mid) / range_size
    else:
        # Out of range
        if rhythm_features.tempo_bpm < tempo_range[0]:
            distance = tempo_range[0] - rhythm_features.tempo_bpm
        else:
            distance = rhythm_features.tempo_bpm - tempo_range[1]
        tempo_score = max(0, 1.0 - distance / 30)
    scores.append(tempo_score)

    # Swing accuracy
    swing_target = dna.rhythm.swing_amount
    swing_diff = abs(rhythm_features.swing_amount - swing_target)
    scores.append(max(0, 1.0 - swing_diff * 2))

    # Kick pattern accuracy
    if rhythm_features.detected_kick_pattern == dna.rhythm.kick_pattern:
        scores.append(1.0)
    else:
        scores.append(0.3)

    # Snare position accuracy
    if rhythm_features.detected_snare_position == dna.rhythm.snare_position:
        scores.append(1.0)
    else:
        scores.append(0.4)

    # Time feel accuracy
    if rhythm_features.detected_time_feel == dna.rhythm.time_feel:
        scores.append(1.0)
    else:
        scores.append(0.3)

    return sum(scores) / len(scores)


def quick_evaluate(
    notes: list[dict],
    tempo_bpm: float,
    target_genre: str,
    ticks_per_beat: int = 480,
) -> dict:
    """
    Quick evaluation of a single generation.

    Args:
        notes: MIDI notes
        tempo_bpm: Track tempo
        target_genre: Target genre
        ticks_per_beat: MIDI resolution

    Returns:
        Dict with evaluation scores
    """
    detector = GenreDetector()
    result = detector.detect(notes, tempo_bpm, ticks_per_beat, target_genre)

    return {
        "predicted_genre": result.primary_genre,
        "confidence": result.confidence,
        "target_similarity": result.dna_similarity,
        "is_correct": result.primary_genre == target_genre,
        "rhythm_accuracy": compute_rhythm_accuracy(
            result.rhythm_features, target_genre
        ),
        "deviations": detector.get_genre_deviations(result, target_genre),
    }
