"""
Identity Invariants

Defines the "Do Not Drift" list - characteristics that must remain
constant across all genres and emotional states to maintain
vocal identity consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from aether.voice.identity.blueprint import VocalIdentity


@dataclass
class InvariantSpec:
    """Specification for a single invariant characteristic."""
    name: str
    description: str
    tolerance: float  # Maximum allowed deviation (0.0-1.0 normalized)
    weight: float = 1.0  # Importance weight for scoring


class IdentityInvariants:
    """
    Manages identity invariants - characteristics that must not change.

    The "Do Not Drift" list ensures vocal consistency across:
    - Different genres
    - Different emotional states
    - Model updates
    """

    INVARIANTS: List[InvariantSpec] = [
        InvariantSpec(
            name="fundamental_timbre",
            description="Core formant relationships preserved within ±5%",
            tolerance=0.05,
            weight=1.5,
        ),
        InvariantSpec(
            name="vowel_signature",
            description="Characteristic 'a' and 'o' shaping maintained",
            tolerance=0.08,
            weight=1.2,
        ),
        InvariantSpec(
            name="vibrato_dna",
            description="Rate 5.2-5.8 Hz, onset delay 180-280ms",
            tolerance=0.10,
            weight=1.0,
        ),
        InvariantSpec(
            name="breath_sound",
            description="Consistent inhale/exhale character",
            tolerance=0.12,
            weight=0.8,
        ),
        InvariantSpec(
            name="transition_smoothness",
            description="No hard register breaks",
            tolerance=0.10,
            weight=1.0,
        ),
        InvariantSpec(
            name="consonant_articulation",
            description="Consistent attack/release profile",
            tolerance=0.08,
            weight=1.1,
        ),
        InvariantSpec(
            name="sibilance_character",
            description="Controlled 's' and 'sh' brightness",
            tolerance=0.10,
            weight=0.9,
        ),
    ]

    def __init__(self, reference_identity: VocalIdentity):
        """
        Initialize with a reference identity to compare against.

        Args:
            reference_identity: The canonical vocal identity definition
        """
        self.reference = reference_identity
        self.reference_vector = reference_identity.get_identity_vector()

    def extract_invariant_values(self, identity: VocalIdentity) -> Dict[str, float]:
        """
        Extract invariant values from an identity for comparison.
        """
        return {
            "fundamental_timbre": self._compute_timbre_signature(identity),
            "vowel_signature": self._compute_vowel_signature(identity),
            "vibrato_dna": self._compute_vibrato_signature(identity),
            "breath_sound": identity.timbre.breathiness,
            "transition_smoothness": identity.transition_smoothness,
            "consonant_articulation": identity.consonant_clarity,
            "sibilance_character": identity.sibilance_brightness,
        }

    def _compute_timbre_signature(self, identity: VocalIdentity) -> float:
        """Compute a normalized timbre signature value."""
        formants = identity.formants.to_array()
        ref_formants = self.reference.formants.to_array()
        # Normalized similarity (1.0 = identical)
        diff = np.abs(formants - ref_formants) / ref_formants
        return 1.0 - np.mean(diff)

    def _compute_vowel_signature(self, identity: VocalIdentity) -> float:
        """Compute vowel shaping signature."""
        # Based on F1/F2 relationship which defines vowel space
        f1 = np.mean(identity.formants.f1_range)
        f2 = np.mean(identity.formants.f2_range)
        ref_f1 = np.mean(self.reference.formants.f1_range)
        ref_f2 = np.mean(self.reference.formants.f2_range)

        f1_diff = abs(f1 - ref_f1) / ref_f1
        f2_diff = abs(f2 - ref_f2) / ref_f2
        return 1.0 - (f1_diff + f2_diff) / 2

    def _compute_vibrato_signature(self, identity: VocalIdentity) -> float:
        """Compute vibrato characteristic signature."""
        rate = np.mean(identity.vibrato_rate_hz)
        onset = np.mean(identity.vibrato_onset_delay_ms)
        ref_rate = np.mean(self.reference.vibrato_rate_hz)
        ref_onset = np.mean(self.reference.vibrato_onset_delay_ms)

        rate_diff = abs(rate - ref_rate) / ref_rate
        onset_diff = abs(onset - ref_onset) / ref_onset
        return 1.0 - (rate_diff + onset_diff) / 2

    def check_invariants(
        self,
        identity: VocalIdentity
    ) -> Tuple[bool, Dict[str, Tuple[float, bool]]]:
        """
        Check if an identity maintains all invariants.

        Returns:
            Tuple of (all_passed, details_dict)
            details_dict maps invariant name to (deviation, passed)
        """
        values = self.extract_invariant_values(identity)
        ref_values = self.extract_invariant_values(self.reference)

        results = {}
        all_passed = True

        for invariant in self.INVARIANTS:
            current = values[invariant.name]
            reference = ref_values[invariant.name]

            # Calculate deviation
            if reference != 0:
                deviation = abs(current - reference) / reference
            else:
                deviation = abs(current - reference)

            passed = deviation <= invariant.tolerance
            results[invariant.name] = (deviation, passed)

            if not passed:
                all_passed = False

        return all_passed, results

    def compute_invariant_score(self, identity: VocalIdentity) -> float:
        """
        Compute a weighted invariant preservation score.

        Returns:
            Score from 0.0 to 1.0, where 1.0 means perfect preservation
        """
        _, results = self.check_invariants(identity)

        total_weight = sum(inv.weight for inv in self.INVARIANTS)
        weighted_score = 0.0

        for invariant in self.INVARIANTS:
            deviation, _ = results[invariant.name]
            # Convert deviation to score (0 deviation = 1.0 score)
            invariant_score = max(0, 1.0 - deviation / invariant.tolerance)
            weighted_score += invariant_score * invariant.weight

        return weighted_score / total_weight


@dataclass
class FlexibleTrait:
    """Specification for a trait that can adapt by genre."""
    name: str
    description: str
    min_value: float
    max_value: float
    default_value: float


class ControlledFlexibility:
    """
    Manages traits that can adapt by genre while preserving identity.

    These traits are allowed to vary within defined ranges based on
    genre requirements.
    """

    FLEXIBLE_TRAITS: List[FlexibleTrait] = [
        FlexibleTrait(
            name="vibrato_depth",
            description="Vibrato pitch deviation in semitones",
            min_value=0.1,
            max_value=0.8,
            default_value=0.4,
        ),
        FlexibleTrait(
            name="breathiness",
            description="Amount of air in tone",
            min_value=0.1,
            max_value=0.5,
            default_value=0.25,
        ),
        FlexibleTrait(
            name="grit_activation",
            description="Raspiness level",
            min_value=0.0,
            max_value=0.6,
            default_value=0.15,
        ),
        FlexibleTrait(
            name="dynamics_range",
            description="Dynamic range in dB",
            min_value=6.0,
            max_value=18.0,
            default_value=12.0,
        ),
        FlexibleTrait(
            name="rhythmic_pocket",
            description="Timing offset from grid in ms",
            min_value=-30.0,
            max_value=20.0,
            default_value=0.0,
        ),
        FlexibleTrait(
            name="register_balance",
            description="Chest/head voice ratio (0=all head, 1=all chest)",
            min_value=0.3,
            max_value=0.7,
            default_value=0.5,
        ),
        FlexibleTrait(
            name="ornamentation_density",
            description="Frequency of ornamental additions",
            min_value=0.0,
            max_value=0.7,
            default_value=0.2,
        ),
    ]

    @classmethod
    def get_trait(cls, name: str) -> Optional[FlexibleTrait]:
        """Get a flexible trait by name."""
        for trait in cls.FLEXIBLE_TRAITS:
            if trait.name == name:
                return trait
        return None

    @classmethod
    def validate_value(cls, name: str, value: float) -> bool:
        """Check if a value is within the allowed range for a trait."""
        trait = cls.get_trait(name)
        if trait is None:
            return False
        return trait.min_value <= value <= trait.max_value

    @classmethod
    def clamp_value(cls, name: str, value: float) -> float:
        """Clamp a value to the allowed range for a trait."""
        trait = cls.get_trait(name)
        if trait is None:
            return value
        return max(trait.min_value, min(trait.max_value, value))


# Pre-configured invariants for AVU-1 identity
# Note: This is a lazy-loaded singleton to avoid circular imports
_avu1_invariants_instance: Optional[IdentityInvariants] = None


def get_avu1_invariants() -> IdentityInvariants:
    """Get the AVU-1 identity invariants (lazy-loaded)."""
    global _avu1_invariants_instance
    if _avu1_invariants_instance is None:
        from aether.voice.identity.blueprint import AVU1Identity
        _avu1_invariants_instance = IdentityInvariants(AVU1Identity)
    return _avu1_invariants_instance


class _AVU1InvariantsAccessor:
    """Lazy accessor for AVU1_INVARIANTS to avoid circular imports."""

    @property
    def invariants(self) -> List[InvariantSpec]:
        return get_avu1_invariants().INVARIANTS

    def validate(self, identity: VocalIdentity) -> List:
        return get_avu1_invariants().check_invariants(identity)[1]

    def __getattr__(self, name):
        return getattr(get_avu1_invariants(), name)


AVU1_INVARIANTS = _AVU1InvariantsAccessor()


def validate_identity(
    identity: VocalIdentity,
    reference: VocalIdentity,
    strict: bool = False
) -> Tuple[bool, str]:
    """
    Validate that an identity maintains required invariants.

    Args:
        identity: The identity to validate
        reference: The reference identity to compare against
        strict: If True, require all invariants to pass

    Returns:
        Tuple of (valid, message)
    """
    invariants = IdentityInvariants(reference)
    all_passed, results = invariants.check_invariants(identity)

    if all_passed:
        return True, "All identity invariants preserved"

    failed = [name for name, (_, passed) in results.items() if not passed]

    if strict:
        return False, f"Identity invariants violated: {', '.join(failed)}"

    # In non-strict mode, allow minor violations
    score = invariants.compute_invariant_score(identity)
    if score >= 0.85:
        return True, f"Identity mostly preserved (score: {score:.2f}), minor drift in: {', '.join(failed)}"

    return False, f"Identity drift detected (score: {score:.2f}), violations: {', '.join(failed)}"
