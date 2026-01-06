"""
Emotional Expression Mapping

Maps emotional intent to vocal performance parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class EmotionCategory(Enum):
    """Primary emotion categories."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    LOVE = "love"
    LONGING = "longing"
    HOPE = "hope"
    DESPAIR = "despair"
    TRIUMPH = "triumph"
    VULNERABILITY = "vulnerability"
    NEUTRAL = "neutral"


@dataclass
class EmotionVector:
    """Multi-dimensional emotion representation."""
    valence: float = 0.0  # -1 (negative) to +1 (positive)
    arousal: float = 0.0  # -1 (calm) to +1 (excited)
    dominance: float = 0.0  # -1 (submissive) to +1 (dominant)
    tension: float = 0.0  # 0 (relaxed) to 1 (tense)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.valence, self.arousal, self.dominance, self.tension])

    @classmethod
    def from_category(cls, category: EmotionCategory) -> "EmotionVector":
        """Create vector from emotion category."""
        mappings = {
            EmotionCategory.JOY: cls(0.8, 0.6, 0.3, 0.2),
            EmotionCategory.SADNESS: cls(-0.6, -0.4, -0.5, 0.4),
            EmotionCategory.ANGER: cls(-0.7, 0.8, 0.8, 0.9),
            EmotionCategory.FEAR: cls(-0.8, 0.6, -0.7, 0.8),
            EmotionCategory.LOVE: cls(0.9, 0.3, 0.0, 0.2),
            EmotionCategory.LONGING: cls(-0.2, -0.2, -0.3, 0.5),
            EmotionCategory.HOPE: cls(0.5, 0.3, 0.2, 0.3),
            EmotionCategory.DESPAIR: cls(-0.9, -0.3, -0.8, 0.7),
            EmotionCategory.TRIUMPH: cls(0.9, 0.8, 0.9, 0.4),
            EmotionCategory.VULNERABILITY: cls(-0.1, -0.4, -0.6, 0.5),
            EmotionCategory.NEUTRAL: cls(0.0, 0.0, 0.0, 0.2),
        }
        return mappings.get(category, cls())

    def blend(self, other: "EmotionVector", weight: float = 0.5) -> "EmotionVector":
        """Blend with another emotion vector."""
        return EmotionVector(
            valence=self.valence + (other.valence - self.valence) * weight,
            arousal=self.arousal + (other.arousal - self.arousal) * weight,
            dominance=self.dominance + (other.dominance - self.dominance) * weight,
            tension=self.tension + (other.tension - self.tension) * weight,
        )


@dataclass
class ExpressionParameters:
    """Vocal expression parameters derived from emotion."""
    # Pitch modifications
    pitch_range_scale: float = 1.0  # Expand/contract pitch range
    pitch_center_shift: float = 0.0  # Shift pitch center (semitones)
    vibrato_depth_scale: float = 1.0
    vibrato_rate_scale: float = 1.0

    # Timing modifications
    tempo_variation: float = 0.0  # -1 to +1 (slower to faster)
    timing_looseness: float = 0.0  # 0 to 1 (strict to loose)
    pause_tendency: float = 0.0  # 0 to 1

    # Dynamics
    dynamic_range_scale: float = 1.0
    attack_intensity: float = 0.5  # 0 (soft) to 1 (hard)
    sustain_weight: float = 0.5  # 0 (short) to 1 (long)

    # Timbre
    breathiness: float = 0.0  # Additional breathiness
    brightness: float = 0.0  # Timbre brightness shift
    grit_amount: float = 0.0  # Vocal fry/grit
    nasality_shift: float = 0.0

    # Articulation
    consonant_emphasis: float = 0.0  # -1 to +1
    legato_preference: float = 0.0  # -1 to +1

    # Ornamentation
    ornamentation_frequency: float = 1.0  # Multiplier
    run_complexity_boost: int = 0  # Added complexity


class ExpressionMapper:
    """
    Maps emotional content to vocal expression parameters.

    Takes emotion analysis and produces concrete parameter
    adjustments for the singing engine.
    """

    def __init__(self):
        """Initialize expression mapper."""
        # Base mapping coefficients (from research on vocal expression)
        self._init_mapping_matrices()

    def _init_mapping_matrices(self):
        """Initialize mapping coefficient matrices."""
        # How valence affects parameters
        self.valence_map = {
            "pitch_range_scale": 0.2,  # Happy = wider range
            "vibrato_depth_scale": 0.1,
            "dynamic_range_scale": 0.15,
            "brightness": 0.3,  # Happy = brighter
            "legato_preference": 0.2,  # Happy = more connected
        }

        # How arousal affects parameters
        self.arousal_map = {
            "pitch_range_scale": 0.25,  # Excited = wider range
            "tempo_variation": 0.3,  # Excited = faster
            "attack_intensity": 0.4,  # Excited = harder attacks
            "vibrato_rate_scale": 0.2,  # Excited = faster vibrato
            "consonant_emphasis": 0.3,
            "ornamentation_frequency": 0.2,
        }

        # How dominance affects parameters
        self.dominance_map = {
            "pitch_center_shift": -1.0,  # Dominant = lower pitch
            "dynamic_range_scale": 0.2,
            "attack_intensity": 0.3,
            "grit_amount": 0.2,  # Dominant = more grit
        }

        # How tension affects parameters
        self.tension_map = {
            "pitch_range_scale": -0.1,  # Tense = narrower
            "timing_looseness": -0.2,  # Tense = tighter timing
            "breathiness": 0.2,
            "vibrato_depth_scale": 0.15,
            "pause_tendency": 0.1,
        }

    def map_emotion(
        self,
        emotion: EmotionVector,
        intensity: float = 1.0,
    ) -> ExpressionParameters:
        """
        Map emotion to expression parameters.

        Args:
            emotion: Emotion vector
            intensity: Overall intensity multiplier (0-1)

        Returns:
            Expression parameters
        """
        params = ExpressionParameters()

        # Apply valence effects
        for param, coef in self.valence_map.items():
            delta = emotion.valence * coef * intensity
            current = getattr(params, param)
            setattr(params, param, current + delta)

        # Apply arousal effects
        for param, coef in self.arousal_map.items():
            delta = emotion.arousal * coef * intensity
            current = getattr(params, param)
            setattr(params, param, current + delta)

        # Apply dominance effects
        for param, coef in self.dominance_map.items():
            delta = emotion.dominance * coef * intensity
            current = getattr(params, param)
            setattr(params, param, current + delta)

        # Apply tension effects
        for param, coef in self.tension_map.items():
            delta = emotion.tension * coef * intensity
            current = getattr(params, param)
            setattr(params, param, current + delta)

        # Clamp values to valid ranges
        params = self._clamp_parameters(params)

        return params

    def _clamp_parameters(self, params: ExpressionParameters) -> ExpressionParameters:
        """Clamp parameters to valid ranges."""
        params.pitch_range_scale = np.clip(params.pitch_range_scale, 0.5, 1.5)
        params.pitch_center_shift = np.clip(params.pitch_center_shift, -3.0, 3.0)
        params.vibrato_depth_scale = np.clip(params.vibrato_depth_scale, 0.3, 2.0)
        params.vibrato_rate_scale = np.clip(params.vibrato_rate_scale, 0.7, 1.5)
        params.tempo_variation = np.clip(params.tempo_variation, -0.3, 0.3)
        params.timing_looseness = np.clip(params.timing_looseness, 0.0, 0.5)
        params.pause_tendency = np.clip(params.pause_tendency, 0.0, 0.5)
        params.dynamic_range_scale = np.clip(params.dynamic_range_scale, 0.5, 1.5)
        params.attack_intensity = np.clip(params.attack_intensity, 0.1, 1.0)
        params.sustain_weight = np.clip(params.sustain_weight, 0.2, 1.0)
        params.breathiness = np.clip(params.breathiness, 0.0, 0.5)
        params.brightness = np.clip(params.brightness, -0.5, 0.5)
        params.grit_amount = np.clip(params.grit_amount, 0.0, 0.5)
        params.nasality_shift = np.clip(params.nasality_shift, -0.3, 0.3)
        params.consonant_emphasis = np.clip(params.consonant_emphasis, -0.5, 0.5)
        params.legato_preference = np.clip(params.legato_preference, -0.5, 0.5)
        params.ornamentation_frequency = np.clip(params.ornamentation_frequency, 0.5, 2.0)
        params.run_complexity_boost = int(np.clip(params.run_complexity_boost, -2, 2))

        return params

    def map_category(
        self,
        category: EmotionCategory,
        intensity: float = 1.0,
    ) -> ExpressionParameters:
        """
        Map emotion category to expression parameters.

        Args:
            category: Emotion category
            intensity: Intensity multiplier

        Returns:
            Expression parameters
        """
        emotion = EmotionVector.from_category(category)
        return self.map_emotion(emotion, intensity)

    def map_text_sentiment(
        self,
        text: str,
        base_emotion: Optional[EmotionCategory] = None,
    ) -> ExpressionParameters:
        """
        Map text sentiment to expression parameters.

        Simple keyword-based sentiment analysis.

        Args:
            text: Lyric text
            base_emotion: Optional base emotion to blend with

        Returns:
            Expression parameters
        """
        # Simple keyword matching (in production, use NLP model)
        positive_words = {
            "love", "happy", "joy", "smile", "sun", "bright", "hope",
            "dream", "heart", "dance", "fly", "free", "alive", "beautiful",
        }
        negative_words = {
            "sad", "cry", "tear", "pain", "lost", "dark", "alone",
            "broken", "cold", "fear", "hurt", "gone", "fall", "empty",
        }
        intense_words = {
            "fire", "burn", "scream", "forever", "never", "always",
            "everything", "nothing", "die", "kill", "hate", "rage",
        }

        words = text.lower().split()

        # Calculate sentiment scores
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)
        intense_count = sum(1 for w in words if w in intense_words)

        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            valence = 0.0
        else:
            valence = (positive_count - negative_count) / total_sentiment_words

        arousal = min(1.0, intense_count * 0.3)

        emotion = EmotionVector(
            valence=valence,
            arousal=arousal,
            dominance=arousal * 0.5,
            tension=abs(valence) * 0.3 + arousal * 0.3,
        )

        # Blend with base emotion if provided
        if base_emotion:
            base_vector = EmotionVector.from_category(base_emotion)
            emotion = emotion.blend(base_vector, 0.5)

        return self.map_emotion(emotion)


class PhraseDynamicsPlanner:
    """
    Plans dynamic contours for musical phrases.
    """

    def __init__(self):
        """Initialize dynamics planner."""
        pass

    def plan_phrase_dynamics(
        self,
        phrase_length_beats: float,
        emotion: EmotionVector,
        phrase_position: str = "middle",  # start, middle, end, climax
    ) -> List[float]:
        """
        Plan dynamic levels across a phrase.

        Args:
            phrase_length_beats: Length of phrase in beats
            emotion: Emotion for the phrase
            phrase_position: Position in song structure

        Returns:
            List of dynamic values (0-1) per beat
        """
        num_beats = max(1, int(phrase_length_beats))
        t = np.linspace(0, 1, num_beats)

        # Base contour shape based on position
        if phrase_position == "start":
            # Build up
            base = 0.4 + 0.3 * t
        elif phrase_position == "end":
            # Wind down
            base = 0.7 - 0.2 * t
        elif phrase_position == "climax":
            # High throughout with slight arc
            base = 0.8 + 0.15 * np.sin(t * np.pi)
        else:
            # Natural arc
            base = 0.5 + 0.25 * np.sin(t * np.pi)

        # Modify based on emotion
        arousal_boost = emotion.arousal * 0.15
        tension_variation = emotion.tension * 0.1 * np.sin(t * np.pi * 4)

        dynamics = base + arousal_boost + tension_variation

        # Clamp to valid range
        dynamics = np.clip(dynamics, 0.2, 1.0)

        return dynamics.tolist()

    def plan_crescendo(
        self,
        start_level: float,
        end_level: float,
        length_beats: float,
        curve: str = "linear",
    ) -> List[float]:
        """
        Plan a crescendo or decrescendo.

        Args:
            start_level: Starting dynamic level (0-1)
            end_level: Ending dynamic level (0-1)
            length_beats: Duration in beats
            curve: Curve shape (linear, exponential, logarithmic)

        Returns:
            List of dynamic values
        """
        num_beats = max(1, int(length_beats))
        t = np.linspace(0, 1, num_beats)

        if curve == "exponential":
            curve_values = t ** 2
        elif curve == "logarithmic":
            curve_values = np.sqrt(t)
        else:
            curve_values = t

        dynamics = start_level + (end_level - start_level) * curve_values
        return dynamics.tolist()


class ExpressionContourGenerator:
    """
    Generates continuous expression contours for phrases.
    """

    def __init__(self):
        """Initialize contour generator."""
        self.mapper = ExpressionMapper()
        self.dynamics_planner = PhraseDynamicsPlanner()

    def generate_expression_contour(
        self,
        phrase_beats: float,
        start_emotion: EmotionVector,
        end_emotion: Optional[EmotionVector] = None,
        frame_rate: float = 10.0,  # Frames per beat
    ) -> Dict[str, np.ndarray]:
        """
        Generate expression contours for a phrase.

        Args:
            phrase_beats: Phrase length in beats
            start_emotion: Emotion at phrase start
            end_emotion: Emotion at phrase end (or same as start)
            frame_rate: Output frame rate

        Returns:
            Dict of parameter contours
        """
        end_emotion = end_emotion or start_emotion
        num_frames = max(1, int(phrase_beats * frame_rate))

        # Interpolate emotion across phrase
        t = np.linspace(0, 1, num_frames)

        contours = {
            "pitch_range_scale": [],
            "vibrato_depth": [],
            "vibrato_rate": [],
            "dynamics": [],
            "breathiness": [],
            "brightness": [],
        }

        for i in range(num_frames):
            # Blend emotion
            weight = t[i]
            emotion = start_emotion.blend(end_emotion, weight)

            # Get parameters for this emotion
            params = self.mapper.map_emotion(emotion)

            # Store in contours
            contours["pitch_range_scale"].append(params.pitch_range_scale)
            contours["vibrato_depth"].append(params.vibrato_depth_scale)
            contours["vibrato_rate"].append(params.vibrato_rate_scale)
            contours["dynamics"].append(params.attack_intensity)
            contours["breathiness"].append(params.breathiness)
            contours["brightness"].append(params.brightness)

        # Convert to numpy arrays and smooth
        for key in contours:
            contours[key] = self._smooth_contour(np.array(contours[key]))

        return contours

    def _smooth_contour(
        self,
        contour: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """Smooth a contour with moving average."""
        if len(contour) < window_size:
            return contour

        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(contour, kernel, mode='same')

        return smoothed
