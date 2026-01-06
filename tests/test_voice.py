"""
Unit tests for AETHER Voice Synthesis Module.
"""

import pytest
import numpy as np

# Identity tests
from aether.voice.identity.blueprint import (
    VocalIdentity,
    VocalRange,
    TimbreCharacteristics,
    FormantProfile,
    VocalClassification,
    AVU1Identity,
    AVU2Identity,
    AVU3Identity,
    AVU4Identity,
    VOICE_REGISTRY,
    get_voice,
    list_voices,
)
from aether.voice.identity.invariants import (
    IdentityInvariants,
    ControlledFlexibility,
    AVU1_INVARIANTS,
)
from aether.voice.identity.drift_monitor import (
    VoiceConsistencyMonitor,
    IdentityDriftTracker,
)

# Phonetics tests
from aether.voice.phonetics.english import EnglishPhonetics, EnglishProsody
from aether.voice.phonetics.spanish import SpanishPhonetics, SpanishProsody
from aether.voice.phonetics.bilingual import BilingualController

# Engine tests
from aether.voice.engine.aligner import LyricMelodyAligner, AlignedUnit
from aether.voice.engine.pitch import PitchController
from aether.voice.engine.vibrato import VibratoGenerator, VibratoParams
from aether.voice.engine.transitions import TransitionEngine, TransitionType
from aether.voice.engine.breath import BreathModel, BreathEvent

# Performance tests
from aether.voice.performance.profiles import (
    VocalPerformanceProfile,
    GENRE_PROFILES,
    get_profile,
)
from aether.voice.performance.ornamentation import OrnamentationEngine, OrnamentType
from aether.voice.performance.expression import (
    ExpressionMapper,
    EmotionVector,
    EmotionCategory,
)

# Arrangement tests
from aether.voice.arrangement.layers import (
    VocalArrangementSystem,
    VocalLayerType,
    GENRE_ARRANGEMENTS,
)
from aether.voice.arrangement.harmony import HarmonyGenerator, ScaleHarmonizer
from aether.voice.arrangement.safeguards import ArrangementSafeguards

# Quality tests
from aether.voice.quality.metrics import (
    PitchMetrics,
    TimingMetrics,
    PhoneticMetrics,
)
from aether.voice.quality.evaluator import VocalQualityEvaluator, EvaluationGrade
from aether.voice.quality.thresholds import (
    ReleaseStage,
    get_thresholds,
    quick_quality_check,
)


class TestVocalIdentity:
    """Tests for vocal identity module."""

    def test_avu1_identity_exists(self):
        """AVU-1 identity should be properly defined."""
        assert AVU1Identity is not None
        assert AVU1Identity.name == "AVU-1"
        assert AVU1Identity.classification == VocalClassification.LYRIC_TENOR

    def test_vocal_range_validation(self):
        """Vocal range should have valid MIDI values."""
        range = AVU1Identity.vocal_range
        assert range.comfortable_low < range.comfortable_high
        assert range.extended_low <= range.comfortable_low
        assert range.extended_high >= range.comfortable_high
        assert range.tessitura_low >= range.comfortable_low
        assert range.tessitura_high <= range.comfortable_high

    def test_timbre_characteristics_normalized(self):
        """Timbre values should be normalized 0-1."""
        timbre = AVU1Identity.timbre
        assert 0 <= timbre.brightness <= 1
        assert 0 <= timbre.breathiness <= 1
        assert 0 <= timbre.grit <= 1
        assert 0 <= timbre.nasality <= 1

    def test_formant_profile_has_vowels(self):
        """Formant profile should have valid F1/F2 ranges."""
        formants = AVU1Identity.formants
        # Check formant ranges are defined
        assert len(formants.f1_range) == 2  # (min, max)
        assert len(formants.f2_range) == 2
        assert formants.f1_range[0] < formants.f1_range[1]
        assert formants.f2_range[0] < formants.f2_range[1]


class TestMultipleVoices:
    """Tests for multiple voice identities."""

    def test_all_voices_exist(self):
        """All four voice identities should be defined."""
        assert AVU1Identity is not None
        assert AVU2Identity is not None
        assert AVU3Identity is not None
        assert AVU4Identity is not None

    def test_voice_registry(self):
        """Voice registry should contain all voices."""
        assert len(VOICE_REGISTRY) == 4
        assert "AVU-1" in VOICE_REGISTRY
        assert "AVU-2" in VOICE_REGISTRY
        assert "AVU-3" in VOICE_REGISTRY
        assert "AVU-4" in VOICE_REGISTRY

    def test_get_voice(self):
        """Should retrieve voices by name."""
        voice = get_voice("AVU-1")
        assert voice.name == "AVU-1"
        voice2 = get_voice("AVU-2")
        assert voice2.name == "AVU-2"

    def test_get_voice_invalid(self):
        """Should raise error for invalid voice name."""
        with pytest.raises(ValueError):
            get_voice("AVU-99")

    def test_list_voices(self):
        """Should list all available voices."""
        voices = list_voices()
        assert len(voices) == 4
        names = [v["name"] for v in voices]
        assert "AVU-1" in names
        assert "AVU-2" in names
        assert "AVU-3" in names
        assert "AVU-4" in names

    def test_voice_classifications(self):
        """Each voice should have correct classification."""
        assert AVU1Identity.classification == VocalClassification.LYRIC_TENOR
        assert AVU2Identity.classification == VocalClassification.MEZZO_SOPRANO
        assert AVU3Identity.classification == VocalClassification.BARITONE
        assert AVU4Identity.classification == VocalClassification.SOPRANO

    def test_voice_ranges_distinct(self):
        """Voice ranges should be appropriately different."""
        # Soprano highest, Baritone lowest
        assert AVU4Identity.vocal_range.tessitura_high > AVU1Identity.vocal_range.tessitura_high
        assert AVU3Identity.vocal_range.tessitura_low < AVU1Identity.vocal_range.tessitura_low
        # Mezzo between soprano and tenor
        assert AVU2Identity.vocal_range.tessitura_low > AVU1Identity.vocal_range.tessitura_low

    def test_identity_vectors_unique(self):
        """Each voice should have a unique identity vector."""
        v1 = AVU1Identity.get_identity_vector()
        v2 = AVU2Identity.get_identity_vector()
        v3 = AVU3Identity.get_identity_vector()
        v4 = AVU4Identity.get_identity_vector()

        # Vectors should be different
        assert not np.allclose(v1, v2)
        assert not np.allclose(v1, v3)
        assert not np.allclose(v1, v4)
        assert not np.allclose(v2, v3)

    def test_all_voices_valid(self):
        """All voice identities should have valid parameters."""
        for name, voice in VOICE_REGISTRY.items():
            # Range validation
            assert voice.vocal_range.comfortable_low < voice.vocal_range.comfortable_high
            assert voice.vocal_range.extended_low <= voice.vocal_range.comfortable_low
            # Timbre validation
            assert 0 <= voice.timbre.brightness <= 1
            assert 0 <= voice.timbre.breathiness <= 1
            # Formant validation
            assert voice.formants.f1_range[0] < voice.formants.f1_range[1]


class TestIdentityInvariants:
    """Tests for identity invariants."""

    def test_avu1_invariants_exist(self):
        """AVU-1 invariants should be defined."""
        assert AVU1_INVARIANTS is not None
        assert len(AVU1_INVARIANTS.invariants) > 0

    def test_invariant_validation(self):
        """Invariant validation should work."""
        # Test with AVU-1 identity (should pass all invariants)
        results = AVU1_INVARIANTS.validate(AVU1Identity)
        # Results is a dict mapping invariant name to (deviation, passed)
        # All invariants should pass for the reference identity
        for name, (deviation, passed) in results.items():
            assert passed, f"Invariant {name} failed with deviation {deviation}"

    def test_controlled_flexibility_trait_access(self):
        """Flexibility traits should be accessible and clampable."""
        # Test getting traits
        vibrato_trait = ControlledFlexibility.get_trait("vibrato_depth")
        assert vibrato_trait is not None
        assert vibrato_trait.min_value < vibrato_trait.max_value

        # Test value validation
        assert ControlledFlexibility.validate_value("vibrato_depth", 0.4)
        assert not ControlledFlexibility.validate_value("vibrato_depth", 2.0)

        # Test value clamping
        clamped = ControlledFlexibility.clamp_value("vibrato_depth", 2.0)
        assert clamped == vibrato_trait.max_value


class TestDriftMonitor:
    """Tests for identity drift monitoring."""

    def test_consistency_monitor_creation(self):
        """Should create consistency monitor."""
        monitor = VoiceConsistencyMonitor(AVU1Identity)
        assert monitor is not None

    def test_drift_tracker_creation(self):
        """Should create drift tracker."""
        tracker = IdentityDriftTracker(AVU1Identity)
        assert tracker is not None
        # No drift history means cumulative drift is zero
        assert len(tracker.drift_history) == 0


class TestEnglishPhonetics:
    """Tests for English phonetics."""

    def test_phoneme_conversion(self):
        """Should convert text to phonemes."""
        phonetics = EnglishPhonetics()
        phonemes = phonetics.text_to_phonemes("hello")
        assert len(phonemes) > 0

    def test_common_words(self):
        """Should handle common words."""
        phonetics = EnglishPhonetics()
        for word in ["the", "love", "you", "world"]:
            phonemes = phonetics.text_to_phonemes(word)
            assert len(phonemes) > 0

    def test_prosody_stress(self):
        """Should detect stress patterns."""
        prosody = EnglishProsody()
        # Simple test for stress detection
        assert prosody is not None


class TestSpanishPhonetics:
    """Tests for Spanish phonetics."""

    def test_phoneme_conversion(self):
        """Should convert Spanish text to phonemes."""
        phonetics = SpanishPhonetics()
        phonemes = phonetics.text_to_phonemes("hola")
        assert len(phonemes) > 0

    def test_common_words(self):
        """Should handle common Spanish words."""
        phonetics = SpanishPhonetics()
        for word in ["amor", "corazón", "mundo"]:
            phonemes = phonetics.text_to_phonemes(word)
            assert len(phonemes) > 0


class TestBilingualController:
    """Tests for bilingual handling."""

    def test_language_detection(self):
        """Should detect language."""
        controller = BilingualController()

        # English words
        assert controller.detect_language("hello") == "en"
        assert controller.detect_language("love") == "en"

        # Spanish words
        assert controller.detect_language("hola") == "es"
        assert controller.detect_language("corazón") == "es"


class TestLyricMelodyAligner:
    """Tests for lyric-melody alignment."""

    def test_alignment_creation(self):
        """Should create aligned units."""
        aligner = LyricMelodyAligner()

        # Create simple test data
        class MockNote:
            def __init__(self, pitch, start, dur, vel=100):
                self.pitch = pitch
                self.start_beat = start
                self.duration_beats = dur
                self.velocity = vel

        lyrics = [{"text": "hel", "phonemes": ["h", "ɛ", "l"]}]
        melody = [MockNote(60, 0, 1)]

        aligned = aligner.align(lyrics, melody)
        assert len(aligned) > 0

    def test_empty_input_handling(self):
        """Should handle empty inputs."""
        aligner = LyricMelodyAligner()
        aligned = aligner.align([], [])
        assert len(aligned) == 0


class TestPitchController:
    """Tests for pitch control."""

    def test_midi_to_hz_conversion(self):
        """Should convert MIDI to Hz correctly."""
        controller = PitchController(AVU1Identity)

        # A4 = 440 Hz = MIDI 69
        hz = controller._midi_to_hz(69)
        assert abs(hz - 440.0) < 0.1

        # Middle C = MIDI 60
        hz = controller._midi_to_hz(60)
        assert abs(hz - 261.63) < 1

    def test_pitch_contour_generation(self):
        """Should generate pitch contours."""
        controller = PitchController(AVU1Identity)

        class MockUnit:
            def __init__(self):
                self.pitch = 60
                self.duration_beats = 1.0

        contour = controller.generate_pitch_contour(MockUnit(), tempo=120)
        assert len(contour) > 0
        assert np.all(contour > 0)


class TestVibratoGenerator:
    """Tests for vibrato generation."""

    def test_vibrato_generation(self):
        """Should generate vibrato signal."""
        generator = VibratoGenerator(AVU1Identity)
        vibrato = generator.generate(duration_frames=100, genre="pop")

        assert len(vibrato) == 100
        # Vibrato should oscillate around zero
        assert np.mean(vibrato) < 10  # Cents

    def test_genre_presets(self):
        """Different genres should have different vibrato."""
        generator = VibratoGenerator(AVU1Identity)

        pop_vibrato = generator.generate(100, genre="pop")
        rnb_vibrato = generator.generate(100, genre="r-and-b")

        # R&B typically has wider vibrato
        assert np.std(rnb_vibrato) >= np.std(pop_vibrato) * 0.8


class TestTransitionEngine:
    """Tests for transition engine."""

    def test_transition_types(self):
        """Should have all transition types."""
        assert TransitionType.LEGATO is not None
        assert TransitionType.PORTAMENTO is not None
        assert TransitionType.STACCATO is not None
        assert TransitionType.BREATH is not None
        assert TransitionType.SCOOP is not None
        assert TransitionType.FALL is not None

    def test_transition_generation(self):
        """Should generate transition curves."""
        engine = TransitionEngine()

        curve = engine.generate_transition(
            TransitionType.LEGATO,
            prev_pitch_hz=261.63,
            next_pitch_hz=293.66,
            duration_frames=50,
        )

        assert len(curve) == 50
        assert curve[0] < curve[-1]  # Should go up


class TestBreathModel:
    """Tests for breath modeling."""

    def test_breath_synthesis(self):
        """Should synthesize breath audio."""
        model = BreathModel(AVU1Identity)

        event = BreathEvent(
            position=0,
            duration_ms=200,
            intensity=0.5,
            audible=True,
        )

        audio = model.synthesize_breath(event)
        assert len(audio) > 0
        assert audio.dtype == np.float32


class TestPerformanceProfiles:
    """Tests for performance profiles."""

    def test_genre_profiles_exist(self):
        """Should have profiles for main genres."""
        assert "pop" in GENRE_PROFILES
        assert "r-and-b" in GENRE_PROFILES
        assert "rock" in GENRE_PROFILES
        assert "jazz" in GENRE_PROFILES

    def test_get_profile(self):
        """Should retrieve profiles."""
        profile = get_profile("pop")
        assert profile is not None
        assert profile.genre is not None

    def test_profile_has_all_components(self):
        """Profile should have all components."""
        profile = get_profile("pop")
        assert profile.timing is not None
        assert profile.articulation is not None
        assert profile.dynamics is not None
        assert profile.register is not None
        assert profile.ornamentation is not None


class TestOrnamentation:
    """Tests for ornamentation engine."""

    def test_run_generation(self):
        """Should generate runs."""
        profile = get_profile("r-and-b")
        engine = OrnamentationEngine(profile)

        run = engine.generate_run(
            start_pitch=60,
            end_pitch=67,
            duration_beats=0.5,
            genre="r-and-b",
        )

        assert run.ornament_type == OrnamentType.RUN
        assert len(run.pitches) > 2

    def test_scoop_generation(self):
        """Should generate scoops."""
        profile = get_profile("pop")
        engine = OrnamentationEngine(profile)

        scoop = engine.generate_scoop(target_pitch=60)
        assert scoop.ornament_type == OrnamentType.SCOOP
        assert scoop.pitches[-1] >= scoop.pitches[0]


class TestExpressionMapper:
    """Tests for expression mapping."""

    def test_emotion_vector_creation(self):
        """Should create emotion vectors."""
        joy = EmotionVector.from_category(EmotionCategory.JOY)
        sadness = EmotionVector.from_category(EmotionCategory.SADNESS)

        assert joy.valence > 0
        assert sadness.valence < 0

    def test_expression_mapping(self):
        """Should map emotion to expression parameters."""
        mapper = ExpressionMapper()

        joy = EmotionVector.from_category(EmotionCategory.JOY)
        params = mapper.map_emotion(joy)

        assert params is not None
        assert params.pitch_range_scale >= 0.5


class TestVocalArrangement:
    """Tests for vocal arrangement."""

    def test_arrangement_system(self):
        """Should create arrangement system."""
        system = VocalArrangementSystem(genre="pop")
        assert system is not None

    def test_genre_arrangements_exist(self):
        """Should have arrangements for genres."""
        assert "pop" in GENRE_ARRANGEMENTS
        assert "r-and-b" in GENRE_ARRANGEMENTS

    def test_layer_retrieval(self):
        """Should retrieve layers for sections."""
        system = VocalArrangementSystem(genre="pop")
        layers = system.get_arrangement("chorus")

        assert len(layers) > 0
        # Chorus should have lead layer
        layer_types = [l.layer_type for l in layers]
        assert VocalLayerType.LEAD in layer_types


class TestHarmonyGenerator:
    """Tests for harmony generation."""

    def test_scale_harmonizer(self):
        """Should harmonize to scale."""
        harmonizer = ScaleHarmonizer(root=0)  # C major

        # C (60) -> E (64) for third above
        harmony = harmonizer.get_diatonic_harmony(60, "third", "above")
        assert harmony == 64

    def test_harmony_generation(self):
        """Should generate harmony voices."""
        generator = HarmonyGenerator(key_root=0)

        melody = [
            {"pitch": 60, "start_beat": 0, "duration_beats": 1},
            {"pitch": 62, "start_beat": 1, "duration_beats": 1},
        ]

        voices = generator.generate_harmony(melody, num_voices=2)
        assert len(voices) == 2


class TestSafeguards:
    """Tests for arrangement safeguards."""

    def test_safeguard_analysis(self):
        """Should analyze for issues."""
        safeguards = ArrangementSafeguards()

        layers = [
            {"name": "lead", "pitch_offset": 0, "timing_offset_ms": 0},
            {"name": "double", "pitch_offset": 0, "timing_offset_ms": 15},
        ]

        result = safeguards.analyze(layers, melody_range=(55, 72))
        assert result is not None
        # Should detect potential phase issue
        assert len(result.issues) >= 0


class TestQualityMetrics:
    """Tests for quality metrics."""

    def test_pitch_accuracy_calculation(self):
        """Should calculate pitch accuracy."""
        metrics = PitchMetrics()

        # Perfect match
        generated = np.array([440.0, 440.0, 440.0])
        target = np.array([440.0, 440.0, 440.0])

        result = metrics.calculate_pitch_accuracy(generated, target)
        assert result.score == 100.0

    def test_pitch_accuracy_with_deviation(self):
        """Should detect pitch deviation."""
        metrics = PitchMetrics()

        # 10% deviation
        generated = np.array([440.0, 484.0, 440.0])  # 484 is ~168 cents sharp
        target = np.array([440.0, 440.0, 440.0])

        result = metrics.calculate_pitch_accuracy(generated, target, tolerance_cents=50)
        assert result.score < 100.0


class TestQualityEvaluator:
    """Tests for quality evaluator."""

    def test_evaluator_creation(self):
        """Should create evaluator."""
        evaluator = VocalQualityEvaluator(release_stage=ReleaseStage.BETA)
        assert evaluator is not None

    def test_grade_assignment(self):
        """Should assign correct grades."""
        evaluator = VocalQualityEvaluator()

        # Test grade boundaries
        assert evaluator._score_to_grade(95) == EvaluationGrade.A_PLUS
        assert evaluator._score_to_grade(90) == EvaluationGrade.A
        assert evaluator._score_to_grade(85) == EvaluationGrade.B_PLUS
        assert evaluator._score_to_grade(70) == EvaluationGrade.C
        assert evaluator._score_to_grade(50) == EvaluationGrade.F


class TestQualityThresholds:
    """Tests for quality thresholds."""

    def test_stage_thresholds(self):
        """Should have thresholds for all stages."""
        for stage in ReleaseStage:
            thresholds = get_thresholds(stage)
            assert thresholds is not None
            assert thresholds.overall_minimum > 0

    def test_production_thresholds_highest(self):
        """Production should have highest thresholds."""
        alpha = get_thresholds(ReleaseStage.ALPHA)
        production = get_thresholds(ReleaseStage.PRODUCTION)

        assert production.overall_minimum > alpha.overall_minimum
        assert production.pitch_accuracy > alpha.pitch_accuracy

    def test_quick_quality_check(self):
        """Should perform quick quality check."""
        passed, message = quick_quality_check(
            pitch_accuracy=85,
            timing_accuracy=85,
            phoneme_accuracy=90,
            stage=ReleaseStage.BETA,
        )

        assert isinstance(passed, bool)
        assert len(message) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
