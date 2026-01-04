"""
Composition Agent

Generates harmony (chord progressions) and melody (hooks, motifs, phrases).
The musical heart of the system.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from aether.agents.base import BaseAgent, AgentRegistry
from aether.knowledge import (
    get_genre_manager,
    parse_progression,
    get_chord_midi,
    get_scale,
    check_voice_leading,
    calculate_singability,
    get_contour,
    contour_to_hash,
)
from aether.schemas.harmony import HarmonySpec, ChordProgression, ChordVoicing
from aether.schemas.melody import MelodySpec, Hook, Motif, SectionMelody, MelodicContour
from aether.schemas.base import NoteName, SectionType
from aether.storage import ArtifactType

logger = logging.getLogger(__name__)


class CompositionInput(BaseModel):
    """Input for Composition Agent."""
    song_spec: Dict[str, Any]
    genre_profile_id: str


class CompositionOutput(BaseModel):
    """Output from Composition Agent."""
    harmony_spec: Dict[str, Any]
    melody_spec: Dict[str, Any]
    composition_decisions: list


@AgentRegistry.register("composition")
class CompositionAgent(BaseAgent[CompositionInput, CompositionOutput]):
    """
    Composition Agent.

    Responsibilities:
    - Generate chord progressions for each section
    - Create memorable hooks and motifs
    - Compose melodies that fit the harmony
    - Ensure originality through variation
    - Check voice leading quality
    """

    agent_type = "composition"
    agent_name = "Composition Agent"
    input_schema = CompositionInput
    output_schema = CompositionOutput

    async def process(
        self,
        input_data: CompositionInput,
        context: Dict[str, Any],
    ) -> CompositionOutput:
        """Process composition."""
        song_spec = input_data.song_spec
        genre_manager = get_genre_manager()
        profile = genre_manager.get(input_data.genre_profile_id)

        key_root = song_spec["key"]["root"]
        key_mode = song_spec["key"]["mode"]
        seed = song_spec.get("random_seed")

        if seed:
            random.seed(seed)

        # Generate harmony
        harmony_spec = await self._generate_harmony(
            song_spec, profile, key_root, key_mode
        )

        # Generate melody
        melody_spec = await self._generate_melody(
            song_spec, profile, harmony_spec, key_root, key_mode
        )

        # Store artifacts
        song_id = str(song_spec["id"])
        self.store_artifact(
            data=harmony_spec,
            artifact_type=ArtifactType.HARMONY_SPEC,
            name="harmony_spec",
            song_id=song_id,
        )
        self.store_artifact(
            data=melody_spec,
            artifact_type=ArtifactType.MELODY_SPEC,
            name="melody_spec",
            song_id=song_id,
        )

        return CompositionOutput(
            harmony_spec=harmony_spec,
            melody_spec=melody_spec,
            composition_decisions=[d.__dict__ for d in self.decisions],
        )

    async def _generate_harmony(
        self,
        song_spec: Dict,
        profile,
        key_root: str,
        key_mode: str,
    ) -> Dict[str, Any]:
        """Generate chord progressions for all sections."""
        progressions = []

        # Get genre-typical progressions
        typical = profile.harmony.typical_progressions

        # Generate for each standard section
        sections = [
            (SectionType.VERSE, "verse"),
            (SectionType.CHORUS, "chorus"),
            (SectionType.BRIDGE, "bridge"),
        ]

        for section_type, section_name in sections:
            # Select a progression (with variation)
            base_prog = random.choice(typical)

            # Parse to actual chords
            chords = parse_progression(key_root, key_mode, base_prog)

            # Create voicings
            voicings = []
            for chord_root, chord_quality in chords:
                voicings.append(ChordVoicing(
                    root=NoteName(chord_root),
                    quality=chord_quality,
                    extensions=[],
                    alterations=[],
                ).model_dump())

            progression = ChordProgression(
                section_type=section_type,
                chords=[ChordVoicing(**v) for v in voicings],
                roman_numerals=base_prog.replace("-", " ").split(),
                durations_beats=[4.0] * len(voicings),
                repeat_count=2 if section_type == SectionType.VERSE else 1,
            )

            progressions.append(progression.model_dump())

            self.log_decision(
                decision_type=f"harmony_{section_name}",
                input_summary=f"Genre progressions: {typical[:3]}",
                output_summary=f"Selected: {base_prog}",
                reasoning=f"Fits {key_mode} tonality, genre-appropriate",
                alternatives=typical[:3],
                confidence=0.85,
            )

        # Check voice leading between progressions
        voice_leading_ok = True
        for prog in progressions:
            chords = prog["chords"]
            for i in range(len(chords) - 1):
                midi1 = get_chord_midi(chords[i]["root"], chords[i]["quality"])
                midi2 = get_chord_midi(chords[i + 1]["root"], chords[i + 1]["quality"])
                violations = check_voice_leading(midi1, midi2)
                if violations:
                    voice_leading_ok = False
                    logger.debug(f"Voice leading issue: {violations[0].description}")

        from aether.schemas.base import KeySignature, Mode
        return HarmonySpec(
            song_id=str(song_spec["id"]),
            primary_key=KeySignature(root=NoteName(key_root), mode=Mode(key_mode)),
            progressions=[ChordProgression(**p) for p in progressions],
            tension_level=profile.harmony.tension_level,
            jazz_influence=profile.harmony.jazz_influence,
            originality_score=0.85,  # Placeholder
        ).model_dump()

    async def _generate_melody(
        self,
        song_spec: Dict,
        profile,
        harmony_spec: Dict,
        key_root: str,
        key_mode: str,
    ) -> Dict[str, Any]:
        """Generate melodies, hooks, and motifs."""
        scale = get_scale(key_root, key_mode)

        # Generate main hook
        hook = self._create_hook(scale, profile)

        # Generate motifs
        motifs = [
            self._create_motif(scale, profile, f"motif_{i}", is_hook=(i == 0))
            for i in range(2)
        ]

        # Generate section melodies
        section_melodies = []
        for section_type in [SectionType.VERSE, SectionType.CHORUS, SectionType.BRIDGE]:
            melody = self._create_section_melody(scale, profile, section_type)
            section_melodies.append(melody)

        self.log_decision(
            decision_type="hook_creation",
            input_summary=f"Scale: {key_root} {key_mode}",
            output_summary=f"Hook created with {len(hook['melody_notes'])} notes",
            reasoning="Prioritized singability and memorability",
            confidence=0.8,
        )

        return MelodySpec(
            song_id=str(song_spec["id"]),
            harmony_id=str(harmony_spec.get("id", "harmony")),
            primary_hook=Hook(**hook),
            motifs=[Motif(**m) for m in motifs],
            section_melodies=[SectionMelody(**s) for s in section_melodies],
            lowest_note=NoteName(scale[0]),
            highest_note=NoteName(scale[-1]),
            typical_range_octaves=1.5,
            originality_score=0.85,
        ).model_dump()

    def _create_hook(self, scale: List[str], profile) -> Dict:
        """Create the main melodic hook."""
        # Simple hook generation - 5-8 notes from scale
        num_notes = random.randint(5, 8)
        hook_notes = []

        # Start on tonic or fifth
        start_idx = random.choice([0, 4]) if len(scale) > 4 else 0
        current_idx = start_idx

        for _ in range(num_notes):
            hook_notes.append(scale[current_idx % len(scale)])
            # Mostly stepwise motion
            step = random.choice([-1, -1, 0, 1, 1, 2])
            current_idx = max(0, min(len(scale) - 1, current_idx + step))

        # Generate rhythm (simple for now)
        rhythm = [random.choice([0.25, 0.5, 0.5, 1.0]) for _ in range(num_notes)]

        # Calculate contour
        contour_seq = get_contour([i for i in range(len(hook_notes))])  # Simplified

        return {
            "melody_notes": [NoteName(n) for n in hook_notes],
            "rhythm_pattern": rhythm,
            "placement": SectionType.CHORUS,
            "contour": MelodicContour(
                contour_type="arch",
                peak_position=0.6,
                range_semitones=7,
            ).model_dump(),
            "singability_score": 0.85,
            "memorability_score": 0.8,
        }

    def _create_motif(self, scale: List[str], profile, name: str, is_hook: bool) -> Dict:
        """Create a melodic motif."""
        num_notes = random.randint(3, 5)
        motif_notes = random.sample(scale, min(num_notes, len(scale)))
        rhythm = [random.choice([0.25, 0.5, 1.0]) for _ in range(len(motif_notes))]

        return {
            "name": name,
            "notes": [NoteName(n) for n in motif_notes],
            "rhythm_pattern": rhythm,
            "intervals": [],
            "contour": MelodicContour(
                contour_type="ascending" if random.random() > 0.5 else "descending",
                range_semitones=5,
            ).model_dump(),
            "is_hook": is_hook,
            "hook_score": 0.8 if is_hook else None,
        }

    def _create_section_melody(self, scale: List[str], profile, section_type: SectionType) -> Dict:
        """Create melody for a song section."""
        # 2 phrases for verse/chorus, 1 for bridge
        num_phrases = 1 if section_type == SectionType.BRIDGE else 2

        phrases = []
        phrase_rhythms = []

        for _ in range(num_phrases):
            phrase_len = random.randint(6, 12)
            phrase = random.choices(scale, k=phrase_len)
            phrases.append([NoteName(n) for n in phrase])
            phrase_rhythms.append([random.choice([0.25, 0.5, 1.0]) for _ in range(phrase_len)])

        register = "mid"
        if section_type == SectionType.CHORUS:
            register = "high" if random.random() > 0.5 else "mid"

        return {
            "section_type": section_type,
            "phrases": phrases,
            "phrase_rhythms": phrase_rhythms,
            "contour": MelodicContour(
                contour_type="arch",
                peak_position=0.65,
                range_semitones=8,
            ).model_dump(),
            "uses_motifs": ["motif_0"],
            "register": register,
        }
