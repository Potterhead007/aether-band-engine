"""
Vocal Arrangement Layers

Multi-layer vocal arrangement system for lead, doubles, harmonies, and ad-libs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class VocalLayerType(Enum):
    """Types of vocal layers."""
    LEAD = "lead"
    DOUBLE = "double"
    HARMONY_HIGH = "harmony_high"
    HARMONY_LOW = "harmony_low"
    HARMONY_THIRD = "harmony_third"
    WHISPER = "whisper"
    AD_LIB = "ad_lib"
    OCTAVE_UP = "octave_up"
    OCTAVE_DOWN = "octave_down"
    CALL = "call"
    RESPONSE = "response"


@dataclass
class VocalLayer:
    """A single vocal layer in the arrangement."""
    layer_type: VocalLayerType
    name: str

    # Pitch offset from lead (semitones, 0 for lead)
    pitch_offset: int = 0

    # Timing offset from lead (ms, positive = behind)
    timing_offset_ms: float = 0.0

    # Mix parameters
    volume_db: float = 0.0
    pan: float = 0.0  # -1 (left) to +1 (right)

    # Character
    breathiness_offset: float = 0.0
    brightness_offset: float = 0.0
    vibrato_scale: float = 1.0

    # Section presence
    active_sections: List[str] = field(default_factory=lambda: ["all"])

    def applies_to_section(self, section: str) -> bool:
        """Check if layer is active for a section."""
        if "all" in self.active_sections:
            return True
        return section.lower() in [s.lower() for s in self.active_sections]


@dataclass
class ArrangementSection:
    """Configuration for a song section's vocal arrangement."""
    name: str  # verse, chorus, bridge, etc.
    layers: List[VocalLayer]
    density: float = 0.5  # 0-1, how "full" the arrangement is
    energy: float = 0.5  # 0-1, energy level


# Default layer configurations
DEFAULT_LAYERS: Dict[VocalLayerType, VocalLayer] = {
    VocalLayerType.LEAD: VocalLayer(
        layer_type=VocalLayerType.LEAD,
        name="Lead Vocal",
        pitch_offset=0,
        timing_offset_ms=0,
        volume_db=0,
        pan=0,
    ),
    VocalLayerType.DOUBLE: VocalLayer(
        layer_type=VocalLayerType.DOUBLE,
        name="Double",
        pitch_offset=0,
        timing_offset_ms=15,  # Slight delay for width
        volume_db=-6,
        pan=0,
        vibrato_scale=0.8,
    ),
    VocalLayerType.HARMONY_HIGH: VocalLayer(
        layer_type=VocalLayerType.HARMONY_HIGH,
        name="High Harmony",
        pitch_offset=4,  # Major 3rd above
        timing_offset_ms=0,
        volume_db=-9,
        pan=0.3,
        breathiness_offset=0.05,
    ),
    VocalLayerType.HARMONY_LOW: VocalLayer(
        layer_type=VocalLayerType.HARMONY_LOW,
        name="Low Harmony",
        pitch_offset=-3,  # Minor 3rd below
        timing_offset_ms=0,
        volume_db=-9,
        pan=-0.3,
        breathiness_offset=0.05,
    ),
    VocalLayerType.HARMONY_THIRD: VocalLayer(
        layer_type=VocalLayerType.HARMONY_THIRD,
        name="Third Harmony",
        pitch_offset=7,  # 5th above
        timing_offset_ms=0,
        volume_db=-12,
        pan=0,
    ),
    VocalLayerType.WHISPER: VocalLayer(
        layer_type=VocalLayerType.WHISPER,
        name="Whisper Layer",
        pitch_offset=0,
        timing_offset_ms=0,
        volume_db=-12,
        pan=0,
        breathiness_offset=0.4,
        brightness_offset=-0.2,
        vibrato_scale=0.3,
    ),
    VocalLayerType.AD_LIB: VocalLayer(
        layer_type=VocalLayerType.AD_LIB,
        name="Ad-lib",
        pitch_offset=0,
        timing_offset_ms=0,
        volume_db=-6,
        pan=0.5,
    ),
    VocalLayerType.OCTAVE_UP: VocalLayer(
        layer_type=VocalLayerType.OCTAVE_UP,
        name="Octave Up",
        pitch_offset=12,
        timing_offset_ms=0,
        volume_db=-12,
        pan=0,
        brightness_offset=0.1,
    ),
    VocalLayerType.OCTAVE_DOWN: VocalLayer(
        layer_type=VocalLayerType.OCTAVE_DOWN,
        name="Octave Down",
        pitch_offset=-12,
        timing_offset_ms=0,
        volume_db=-12,
        pan=0,
        brightness_offset=-0.15,
    ),
}


# Genre-specific arrangement templates
GENRE_ARRANGEMENTS: Dict[str, Dict[str, List[VocalLayerType]]] = {
    "pop": {
        "verse": [VocalLayerType.LEAD],
        "pre_chorus": [VocalLayerType.LEAD, VocalLayerType.DOUBLE],
        "chorus": [
            VocalLayerType.LEAD,
            VocalLayerType.DOUBLE,
            VocalLayerType.HARMONY_HIGH,
            VocalLayerType.HARMONY_LOW,
        ],
        "bridge": [VocalLayerType.LEAD, VocalLayerType.HARMONY_HIGH],
        "outro": [VocalLayerType.LEAD, VocalLayerType.WHISPER],
    },
    "r-and-b": {
        "verse": [VocalLayerType.LEAD, VocalLayerType.WHISPER],
        "pre_chorus": [VocalLayerType.LEAD, VocalLayerType.HARMONY_HIGH],
        "chorus": [
            VocalLayerType.LEAD,
            VocalLayerType.DOUBLE,
            VocalLayerType.HARMONY_HIGH,
            VocalLayerType.HARMONY_LOW,
            VocalLayerType.AD_LIB,
        ],
        "bridge": [VocalLayerType.LEAD, VocalLayerType.HARMONY_HIGH, VocalLayerType.HARMONY_LOW],
        "outro": [VocalLayerType.LEAD, VocalLayerType.AD_LIB],
    },
    "rock": {
        "verse": [VocalLayerType.LEAD],
        "pre_chorus": [VocalLayerType.LEAD, VocalLayerType.DOUBLE],
        "chorus": [
            VocalLayerType.LEAD,
            VocalLayerType.DOUBLE,
            VocalLayerType.HARMONY_HIGH,
        ],
        "bridge": [VocalLayerType.LEAD],
        "outro": [VocalLayerType.LEAD, VocalLayerType.DOUBLE],
    },
    "jazz": {
        "verse": [VocalLayerType.LEAD],
        "chorus": [VocalLayerType.LEAD],
        "bridge": [VocalLayerType.LEAD],
        "outro": [VocalLayerType.LEAD, VocalLayerType.WHISPER],
    },
    "house": {
        "verse": [VocalLayerType.LEAD],
        "chorus": [VocalLayerType.LEAD, VocalLayerType.DOUBLE],
        "breakdown": [VocalLayerType.LEAD, VocalLayerType.WHISPER],
        "drop": [VocalLayerType.LEAD, VocalLayerType.OCTAVE_UP],
    },
    "trap": {
        "verse": [VocalLayerType.LEAD, VocalLayerType.AD_LIB],
        "chorus": [
            VocalLayerType.LEAD,
            VocalLayerType.DOUBLE,
            VocalLayerType.AD_LIB,
        ],
        "bridge": [VocalLayerType.LEAD],
    },
    "ambient": {
        "verse": [VocalLayerType.LEAD, VocalLayerType.WHISPER],
        "chorus": [
            VocalLayerType.LEAD,
            VocalLayerType.WHISPER,
            VocalLayerType.HARMONY_HIGH,
        ],
        "bridge": [VocalLayerType.LEAD, VocalLayerType.HARMONY_HIGH, VocalLayerType.HARMONY_LOW],
    },
}


class VocalArrangementSystem:
    """
    Manages multi-layer vocal arrangements.

    Responsible for:
    - Determining which layers to use per section
    - Configuring layer parameters
    - Managing harmony voice leading
    - Preventing frequency masking
    """

    def __init__(self, genre: str = "pop"):
        """
        Initialize arrangement system.

        Args:
            genre: Genre for default arrangements
        """
        self.genre = genre
        self.custom_layers: Dict[str, VocalLayer] = {}

    def get_arrangement(
        self,
        section: str,
        custom_layers: Optional[List[VocalLayer]] = None,
    ) -> List[VocalLayer]:
        """
        Get vocal arrangement for a section.

        Args:
            section: Section name (verse, chorus, etc.)
            custom_layers: Override with custom layer config

        Returns:
            List of VocalLayer configurations
        """
        if custom_layers:
            return custom_layers

        # Get genre template
        genre_template = GENRE_ARRANGEMENTS.get(
            self.genre,
            GENRE_ARRANGEMENTS["pop"]
        )

        layer_types = genre_template.get(
            section.lower(),
            [VocalLayerType.LEAD]  # Default to just lead
        )

        # Build layer list from types
        layers = []
        for layer_type in layer_types:
            layer = DEFAULT_LAYERS.get(layer_type)
            if layer:
                # Make a copy to avoid mutating defaults
                layer_copy = VocalLayer(
                    layer_type=layer.layer_type,
                    name=layer.name,
                    pitch_offset=layer.pitch_offset,
                    timing_offset_ms=layer.timing_offset_ms,
                    volume_db=layer.volume_db,
                    pan=layer.pan,
                    breathiness_offset=layer.breathiness_offset,
                    brightness_offset=layer.brightness_offset,
                    vibrato_scale=layer.vibrato_scale,
                    active_sections=[section],
                )
                layers.append(layer_copy)

        return layers

    def create_section_config(
        self,
        section_name: str,
        energy_level: float = 0.5,
        harmony_style: str = "thirds",
    ) -> ArrangementSection:
        """
        Create arrangement configuration for a section.

        Args:
            section_name: Name of section
            energy_level: 0-1, affects layer count and volumes
            harmony_style: thirds, fifths, or custom

        Returns:
            ArrangementSection configuration
        """
        # Get base layers
        layers = self.get_arrangement(section_name)

        # Adjust harmony pitches based on style
        if harmony_style == "fifths":
            for layer in layers:
                if layer.layer_type == VocalLayerType.HARMONY_HIGH:
                    layer.pitch_offset = 7  # Perfect 5th
                elif layer.layer_type == VocalLayerType.HARMONY_LOW:
                    layer.pitch_offset = -5  # Perfect 4th below

        # Adjust volumes based on energy
        volume_boost = (energy_level - 0.5) * 6  # +/- 3dB

        for layer in layers:
            if layer.layer_type != VocalLayerType.LEAD:
                layer.volume_db += volume_boost

        return ArrangementSection(
            name=section_name,
            layers=layers,
            density=energy_level,
            energy=energy_level,
        )

    def add_custom_layer(
        self,
        name: str,
        layer: VocalLayer,
    ) -> None:
        """
        Add a custom layer configuration.

        Args:
            name: Layer identifier
            layer: Layer configuration
        """
        self.custom_layers[name] = layer

    def get_layer_mix(
        self,
        layers: List[VocalLayer],
    ) -> Dict[str, Dict[str, float]]:
        """
        Get mixing parameters for all layers.

        Returns:
            Dict mapping layer names to mix parameters
        """
        mix = {}
        for layer in layers:
            mix[layer.name] = {
                "volume_db": layer.volume_db,
                "pan": layer.pan,
                "timing_offset_ms": layer.timing_offset_ms,
            }
        return mix

    def balance_arrangement(
        self,
        layers: List[VocalLayer],
        target_headroom_db: float = -3.0,
    ) -> List[VocalLayer]:
        """
        Balance layer volumes to prevent clipping.

        Args:
            layers: Layers to balance
            target_headroom_db: Target peak level

        Returns:
            Layers with adjusted volumes
        """
        if not layers:
            return layers

        # Calculate approximate peak from summing layers
        # (simplified - real implementation would use true summing)
        linear_sum = sum(
            10 ** (layer.volume_db / 20)
            for layer in layers
        )
        peak_db = 20 * np.log10(linear_sum) if linear_sum > 0 else -60

        # Calculate needed reduction
        reduction = peak_db - target_headroom_db

        if reduction > 0:
            # Apply reduction to all layers
            for layer in layers:
                layer.volume_db -= reduction

        return layers

    def plan_full_arrangement(
        self,
        sections: List[Tuple[str, float, float]],  # (name, start_beat, end_beat)
    ) -> Dict[str, ArrangementSection]:
        """
        Plan arrangement for entire song.

        Args:
            sections: List of (section_name, start, end) tuples

        Returns:
            Dict mapping section names to arrangements
        """
        arrangements = {}

        for section_name, start, end in sections:
            # Determine energy based on section type
            energy_map = {
                "intro": 0.3,
                "verse": 0.5,
                "pre_chorus": 0.65,
                "chorus": 0.8,
                "post_chorus": 0.7,
                "bridge": 0.6,
                "breakdown": 0.4,
                "drop": 0.9,
                "outro": 0.4,
            }

            energy = energy_map.get(section_name.lower(), 0.5)

            arrangements[f"{section_name}_{start}"] = self.create_section_config(
                section_name=section_name,
                energy_level=energy,
            )

        return arrangements


class DynamicArrangementBuilder:
    """
    Builds arrangements dynamically based on song analysis.
    """

    def __init__(self, base_system: VocalArrangementSystem):
        """
        Initialize builder.

        Args:
            base_system: Base arrangement system
        """
        self.system = base_system

    def build_from_energy_curve(
        self,
        energy_curve: List[float],
        section_boundaries: List[Tuple[int, int, str]],
    ) -> List[ArrangementSection]:
        """
        Build arrangement from energy analysis.

        Args:
            energy_curve: Per-beat energy values (0-1)
            section_boundaries: (start_beat, end_beat, section_name) tuples

        Returns:
            List of arrangement sections
        """
        arrangements = []

        for start, end, name in section_boundaries:
            # Get average energy for section
            section_energy = energy_curve[start:end]
            avg_energy = np.mean(section_energy) if section_energy else 0.5

            # Build arrangement
            config = self.system.create_section_config(
                section_name=name,
                energy_level=avg_energy,
            )
            arrangements.append(config)

        return arrangements

    def add_build_up(
        self,
        arrangement: ArrangementSection,
        build_duration_beats: float,
    ) -> List[ArrangementSection]:
        """
        Create gradual build-up to a section.

        Splits section into progressive sub-arrangements.

        Args:
            arrangement: Target arrangement
            build_duration_beats: Duration of build

        Returns:
            List of progressive arrangements
        """
        stages = []
        num_stages = 4

        # Start with fewer layers, add progressively
        all_layers = arrangement.layers.copy()
        lead_layer = [l for l in all_layers if l.layer_type == VocalLayerType.LEAD]

        for i in range(num_stages):
            progress = (i + 1) / num_stages
            num_layers = max(1, int(len(all_layers) * progress))

            # Select layers for this stage
            stage_layers = lead_layer + all_layers[1:num_layers]

            # Adjust volumes for build
            for layer in stage_layers:
                layer.volume_db -= (1 - progress) * 6

            stages.append(ArrangementSection(
                name=f"{arrangement.name}_build_{i}",
                layers=stage_layers,
                density=progress,
                energy=arrangement.energy * progress,
            ))

        return stages
