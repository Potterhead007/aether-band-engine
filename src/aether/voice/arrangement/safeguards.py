"""
Vocal Arrangement Safeguards

Prevents common audio issues in multi-voice arrangements:
- Phase cancellation
- Frequency masking
- Excessive buildup
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class SafeguardType(Enum):
    """Types of safeguards."""
    PHASE = "phase"
    MASKING = "masking"
    BUILDUP = "buildup"
    CLIPPING = "clipping"


@dataclass
class SafeguardIssue:
    """A detected audio issue."""
    issue_type: SafeguardType
    severity: float  # 0-1, 1 being severe
    location: float  # Beat position
    duration: float  # Duration in beats
    description: str
    suggested_fix: str


@dataclass
class SafeguardResult:
    """Result of safeguard analysis."""
    passed: bool
    issues: List[SafeguardIssue]
    corrections_applied: List[str]


class PhaseDetector:
    """
    Detects potential phase cancellation issues.

    Phase issues occur when similar audio signals
    are summed with slight timing differences.
    """

    # Dangerous timing offset ranges (ms)
    DANGER_ZONES = [
        (0, 5),    # Very short delays cause comb filtering
        (10, 25),  # Typical double timing - can cause issues
    ]

    def __init__(self):
        """Initialize phase detector."""
        pass

    def check_phase_issues(
        self,
        layers: List[dict],
    ) -> List[SafeguardIssue]:
        """
        Check for potential phase issues between layers.

        Args:
            layers: Layer configurations with timing offsets

        Returns:
            List of detected issues
        """
        issues = []

        # Check pairs of layers
        for i, layer_a in enumerate(layers):
            for j, layer_b in enumerate(layers):
                if i >= j:
                    continue

                # Check timing offset difference
                offset_a = layer_a.get("timing_offset_ms", 0)
                offset_b = layer_b.get("timing_offset_ms", 0)
                offset_diff = abs(offset_a - offset_b)

                # Check if same pitch content
                same_pitch = layer_a.get("pitch_offset", 0) == layer_b.get("pitch_offset", 0)

                if same_pitch:
                    for low, high in self.DANGER_ZONES:
                        if low <= offset_diff <= high:
                            issues.append(SafeguardIssue(
                                issue_type=SafeguardType.PHASE,
                                severity=0.7 if offset_diff < 5 else 0.4,
                                location=0,
                                duration=0,
                                description=(
                                    f"Potential phase issue between "
                                    f"{layer_a.get('name', 'Layer A')} and "
                                    f"{layer_b.get('name', 'Layer B')}: "
                                    f"{offset_diff:.1f}ms offset"
                                ),
                                suggested_fix=(
                                    f"Increase timing offset to >30ms or "
                                    f"apply complementary EQ"
                                ),
                            ))

        return issues

    def suggest_timing_offset(
        self,
        existing_offsets: List[float],
        sample_rate: int = 48000,
    ) -> float:
        """
        Suggest a safe timing offset for a new layer.

        Args:
            existing_offsets: Offsets of existing layers (ms)
            sample_rate: Audio sample rate

        Returns:
            Suggested offset in ms
        """
        # Safe offsets (avoid comb filtering frequencies)
        safe_offsets = [0, 35, 55, 75, 100]

        for offset in safe_offsets:
            safe = True
            for existing in existing_offsets:
                diff = abs(offset - existing)
                if diff < 30:
                    safe = False
                    break
            if safe:
                return offset

        # If no safe offset, suggest one far from existing
        return max(existing_offsets) + 50 if existing_offsets else 0


class MaskingDetector:
    """
    Detects frequency masking issues.

    Masking occurs when multiple voices occupy
    the same frequency range simultaneously.
    """

    # Vocal frequency ranges (Hz)
    VOCAL_RANGES = {
        "bass": (80, 300),
        "baritone": (100, 400),
        "tenor": (130, 500),
        "alto": (175, 700),
        "soprano": (250, 1000),
    }

    # Fundamental frequency thresholds
    LOW_VOICE = 200  # Hz
    HIGH_VOICE = 400  # Hz

    def __init__(self):
        """Initialize masking detector."""
        pass

    def check_masking(
        self,
        layers: List[dict],
        melody_range: Tuple[int, int],  # MIDI range
    ) -> List[SafeguardIssue]:
        """
        Check for frequency masking issues.

        Args:
            layers: Layer configurations
            melody_range: MIDI pitch range of melody

        Returns:
            List of detected issues
        """
        issues = []

        # Convert melody range to Hz
        melody_low_hz = self._midi_to_hz(melody_range[0])
        melody_high_hz = self._midi_to_hz(melody_range[1])

        # Check each harmony layer
        harmony_layers = [
            l for l in layers
            if l.get("pitch_offset", 0) != 0
        ]

        for layer in harmony_layers:
            offset = layer.get("pitch_offset", 0)

            # Calculate harmony frequency range
            harm_low_hz = self._midi_to_hz(melody_range[0] + offset)
            harm_high_hz = self._midi_to_hz(melody_range[1] + offset)

            # Check overlap
            overlap = self._calculate_overlap(
                (melody_low_hz, melody_high_hz),
                (harm_low_hz, harm_high_hz),
            )

            if overlap > 0.5:  # More than 50% overlap
                issues.append(SafeguardIssue(
                    issue_type=SafeguardType.MASKING,
                    severity=overlap * 0.8,
                    location=0,
                    duration=0,
                    description=(
                        f"High frequency overlap ({overlap*100:.0f}%) "
                        f"between lead and {layer.get('name', 'harmony')}"
                    ),
                    suggested_fix=(
                        "Apply complementary EQ: cut harmony at "
                        f"{int((melody_low_hz + melody_high_hz) / 2)}Hz"
                    ),
                ))

        return issues

    def _midi_to_hz(self, midi: int) -> float:
        """Convert MIDI note to frequency."""
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))

    def _calculate_overlap(
        self,
        range_a: Tuple[float, float],
        range_b: Tuple[float, float],
    ) -> float:
        """Calculate overlap ratio between two ranges."""
        low = max(range_a[0], range_b[0])
        high = min(range_a[1], range_b[1])

        if low >= high:
            return 0.0

        overlap = high - low
        range_a_size = range_a[1] - range_a[0]
        range_b_size = range_b[1] - range_b[0]

        return overlap / min(range_a_size, range_b_size)

    def suggest_eq_curve(
        self,
        lead_range: Tuple[float, float],
        harmony_range: Tuple[float, float],
    ) -> Dict[str, float]:
        """
        Suggest EQ settings to reduce masking.

        Args:
            lead_range: Lead vocal frequency range
            harmony_range: Harmony frequency range

        Returns:
            Dict of EQ parameters
        """
        # Find overlap center
        overlap_low = max(lead_range[0], harmony_range[0])
        overlap_high = min(lead_range[1], harmony_range[1])

        if overlap_low >= overlap_high:
            return {}

        center_freq = np.sqrt(overlap_low * overlap_high)  # Geometric mean

        return {
            "harmony_cut_freq": center_freq,
            "harmony_cut_db": -3.0,
            "harmony_cut_q": 1.5,
            "lead_boost_freq": center_freq,
            "lead_boost_db": 1.5,
            "lead_boost_q": 2.0,
        }


class BuildupDetector:
    """
    Detects excessive low-frequency buildup.

    Common when multiple voices stack in low register.
    """

    # Low frequency threshold
    LOW_FREQ_THRESHOLD = 250  # Hz

    def __init__(self):
        """Initialize buildup detector."""
        pass

    def check_buildup(
        self,
        layers: List[dict],
        melody_range: Tuple[int, int],
    ) -> List[SafeguardIssue]:
        """
        Check for low frequency buildup.

        Args:
            layers: Layer configurations
            melody_range: MIDI pitch range

        Returns:
            List of detected issues
        """
        issues = []

        # Count layers in low register
        low_layer_count = 0

        for layer in layers:
            offset = layer.get("pitch_offset", 0)
            low_pitch = melody_range[0] + offset

            if self._midi_to_hz(low_pitch) < self.LOW_FREQ_THRESHOLD:
                low_layer_count += 1

        if low_layer_count > 2:
            issues.append(SafeguardIssue(
                issue_type=SafeguardType.BUILDUP,
                severity=min(1.0, low_layer_count * 0.3),
                location=0,
                duration=0,
                description=(
                    f"{low_layer_count} layers in low frequency range - "
                    "potential muddy buildup"
                ),
                suggested_fix=(
                    "Apply high-pass filter to harmony layers "
                    "(80-120Hz cutoff) or reduce harmony count"
                ),
            ))

        return issues

    def _midi_to_hz(self, midi: int) -> float:
        """Convert MIDI note to frequency."""
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))


class ArrangementSafeguards:
    """
    Main safeguard system for vocal arrangements.

    Combines all safeguard detectors and provides
    automatic corrections when possible.
    """

    def __init__(self):
        """Initialize safeguard system."""
        self.phase_detector = PhaseDetector()
        self.masking_detector = MaskingDetector()
        self.buildup_detector = BuildupDetector()

    def analyze(
        self,
        layers: List[dict],
        melody_range: Tuple[int, int],
    ) -> SafeguardResult:
        """
        Analyze arrangement for issues.

        Args:
            layers: Layer configurations
            melody_range: MIDI pitch range of melody

        Returns:
            SafeguardResult with any detected issues
        """
        all_issues = []

        # Run all detectors
        all_issues.extend(self.phase_detector.check_phase_issues(layers))
        all_issues.extend(self.masking_detector.check_masking(layers, melody_range))
        all_issues.extend(self.buildup_detector.check_buildup(layers, melody_range))

        # Determine if passed (no severe issues)
        severe_issues = [i for i in all_issues if i.severity > 0.7]
        passed = len(severe_issues) == 0

        return SafeguardResult(
            passed=passed,
            issues=all_issues,
            corrections_applied=[],
        )

    def auto_correct(
        self,
        layers: List[dict],
        melody_range: Tuple[int, int],
    ) -> Tuple[List[dict], SafeguardResult]:
        """
        Analyze and auto-correct arrangement issues.

        Args:
            layers: Layer configurations
            melody_range: MIDI pitch range

        Returns:
            Tuple of (corrected layers, safeguard result)
        """
        # First analysis
        result = self.analyze(layers, melody_range)

        corrections = []
        corrected_layers = [l.copy() for l in layers]

        # Apply phase corrections
        for issue in result.issues:
            if issue.issue_type == SafeguardType.PHASE:
                # Adjust timing offsets
                existing_offsets = [
                    l.get("timing_offset_ms", 0)
                    for l in corrected_layers
                ]
                for layer in corrected_layers:
                    if layer.get("timing_offset_ms", 0) in [0, 15, 20]:
                        new_offset = self.phase_detector.suggest_timing_offset(
                            existing_offsets
                        )
                        layer["timing_offset_ms"] = new_offset
                        corrections.append(
                            f"Adjusted timing offset to {new_offset}ms"
                        )
                        break

            elif issue.issue_type == SafeguardType.BUILDUP:
                # Add high-pass filter suggestion to layers
                for layer in corrected_layers:
                    if layer.get("pitch_offset", 0) < 0:
                        layer["suggested_highpass"] = 100
                        corrections.append(
                            f"Suggested 100Hz highpass for {layer.get('name', 'layer')}"
                        )

        # Re-analyze after corrections
        final_result = self.analyze(corrected_layers, melody_range)
        final_result.corrections_applied = corrections

        return corrected_layers, final_result

    def get_mix_suggestions(
        self,
        layers: List[dict],
        melody_range: Tuple[int, int],
    ) -> Dict[str, dict]:
        """
        Get mixing suggestions for each layer.

        Args:
            layers: Layer configurations
            melody_range: MIDI pitch range

        Returns:
            Dict mapping layer names to suggested mix settings
        """
        suggestions = {}

        for layer in layers:
            name = layer.get("name", "unnamed")
            offset = layer.get("pitch_offset", 0)

            suggestion = {
                "volume_db": layer.get("volume_db", 0),
                "pan": layer.get("pan", 0),
            }

            # Suggest EQ based on pitch
            if offset > 0:
                # Higher harmony - slight high shelf
                suggestion["eq"] = {
                    "high_shelf_freq": 8000,
                    "high_shelf_db": 2,
                }
            elif offset < 0:
                # Lower harmony - cut some low mids
                suggestion["eq"] = {
                    "low_cut_freq": 100,
                    "low_mid_cut_freq": 300,
                    "low_mid_cut_db": -2,
                }

            # Suggest compression
            if layer.get("layer_type") == "lead":
                suggestion["compression"] = {
                    "ratio": 3,
                    "threshold_db": -18,
                    "attack_ms": 10,
                    "release_ms": 100,
                }
            else:
                suggestion["compression"] = {
                    "ratio": 2,
                    "threshold_db": -20,
                    "attack_ms": 15,
                    "release_ms": 150,
                }

            suggestions[name] = suggestion

        return suggestions
