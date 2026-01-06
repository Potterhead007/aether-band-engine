"""
Rhythm Grammar - Genre-specific rhythm pattern generation.

Defines rhythmic grammars for each genre that generate characteristic
drum patterns, ensuring genre authenticity through rule-based generation.
"""

from dataclasses import dataclass, field
from typing import Optional
import random

from aether.genre.dna import (
    GenreDNA,
    KickPattern,
    SnarePosition,
    TimeFeel,
    get_genre_dna,
)


@dataclass
class DrumHit:
    """A single drum hit."""
    instrument: str  # kick, snare, hihat, clap, etc.
    position: float  # Position in beats (0-4 for one bar)
    velocity: int  # 0-127
    duration: float = 0.1  # Duration in beats


@dataclass
class RhythmPattern:
    """A complete rhythm pattern (one bar)."""
    hits: list[DrumHit]
    time_signature: tuple[int, int] = (4, 4)
    swing_amount: float = 0.0

    def get_kick_hits(self) -> list[DrumHit]:
        """Get all kick drum hits."""
        return [h for h in self.hits if h.instrument == "kick"]

    def get_snare_hits(self) -> list[DrumHit]:
        """Get all snare hits."""
        return [h for h in self.hits if h.instrument in ("snare", "clap")]

    def get_hihat_hits(self) -> list[DrumHit]:
        """Get all hi-hat hits."""
        return [h for h in self.hits if h.instrument.startswith("hihat")]

    def to_midi_notes(self, ticks_per_beat: int = 480) -> list[dict]:
        """Convert to MIDI note format."""
        # GM drum map
        drum_map = {
            "kick": 36,
            "snare": 38,
            "clap": 39,
            "rim": 37,
            "hihat_closed": 42,
            "hihat_open": 46,
            "hihat_pedal": 44,
            "tom_high": 50,
            "tom_mid": 47,
            "tom_low": 45,
            "crash": 49,
            "ride": 51,
            "perc": 56,
        }

        notes = []
        for hit in self.hits:
            pitch = drum_map.get(hit.instrument, 38)

            # Apply swing
            position = hit.position
            if self.swing_amount > 0 and (position * 2) % 1 > 0.4:
                # Offbeat 8th notes get pushed back
                position += self.swing_amount * 0.1

            notes.append({
                "pitch": pitch,
                "start_tick": int(position * ticks_per_beat),
                "duration": int(hit.duration * ticks_per_beat),
                "velocity": hit.velocity,
                "channel": 9,  # Drum channel
            })

        return notes


@dataclass
class RhythmGrammar:
    """
    Grammar rules for generating rhythm patterns in a genre.

    Each genre has specific rules for kick, snare, and hi-hat placement
    that create its characteristic groove.
    """
    # Kick rules (required - no defaults)
    kick_positions: list[float]  # Beat positions where kicks can occur
    kick_weights: list[float]  # Probability weights for each position

    # Snare rules (required - no defaults)
    snare_positions: list[float]
    snare_weights: list[float]

    # Optional fields with defaults
    kick_velocity_range: tuple[int, int] = (100, 127)
    snare_velocity_range: tuple[int, int] = (90, 120)
    use_clap: bool = False

    # Hi-hat rules
    hihat_subdivision: int = 8  # 8 = 8th notes, 16 = 16th notes
    hihat_open_positions: list[float] = field(default_factory=list)
    hihat_accent_positions: list[float] = field(default_factory=list)
    hihat_velocity_range: tuple[int, int] = (60, 90)
    hihat_variation: float = 0.1  # Velocity variation

    # Feel
    swing_amount: float = 0.0
    ghost_note_probability: float = 0.0
    fill_probability: float = 0.1


# Genre-specific rhythm grammars
GENRE_RHYTHM_GRAMMAR: dict[str, RhythmGrammar] = {
    "lofi-hip-hop": RhythmGrammar(
        kick_positions=[0.0, 1.75, 2.5],
        kick_weights=[1.0, 0.7, 0.5],
        kick_velocity_range=(80, 100),
        snare_positions=[1.0, 3.0],
        snare_weights=[1.0, 1.0],
        snare_velocity_range=(70, 90),
        hihat_subdivision=8,
        hihat_velocity_range=(40, 70),
        hihat_variation=0.2,
        swing_amount=0.3,
        ghost_note_probability=0.2,
    ),
    "trap": RhythmGrammar(
        kick_positions=[0.0, 0.75, 2.25, 3.5],
        kick_weights=[1.0, 0.6, 0.8, 0.4],
        kick_velocity_range=(110, 127),
        snare_positions=[1.0, 3.0],
        snare_weights=[1.0, 1.0],
        snare_velocity_range=(100, 120),
        use_clap=True,
        hihat_subdivision=16,
        hihat_open_positions=[1.5, 3.5],
        hihat_accent_positions=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        hihat_velocity_range=(70, 110),
        hihat_variation=0.3,
        swing_amount=0.0,
        ghost_note_probability=0.0,
    ),
    "drill": RhythmGrammar(
        kick_positions=[0.0, 0.5, 2.0, 2.75],
        kick_weights=[1.0, 0.5, 0.8, 0.6],
        kick_velocity_range=(115, 127),
        snare_positions=[1.0, 2.5, 3.0],
        snare_weights=[1.0, 0.5, 1.0],
        snare_velocity_range=(105, 125),
        use_clap=True,
        hihat_subdivision=16,
        hihat_velocity_range=(80, 115),
        hihat_variation=0.25,
        swing_amount=0.0,
    ),
    "boom-bap": RhythmGrammar(
        kick_positions=[0.0, 1.5, 2.75],
        kick_weights=[1.0, 0.7, 0.5],
        kick_velocity_range=(95, 115),
        snare_positions=[1.0, 3.0],
        snare_weights=[1.0, 1.0],
        snare_velocity_range=(90, 110),
        hihat_subdivision=8,
        hihat_velocity_range=(50, 80),
        hihat_variation=0.15,
        swing_amount=0.2,
        ghost_note_probability=0.15,
    ),
    "synthwave": RhythmGrammar(
        kick_positions=[0.0, 1.0, 2.0, 3.0],
        kick_weights=[1.0, 1.0, 1.0, 1.0],
        kick_velocity_range=(100, 120),
        snare_positions=[1.0, 3.0],
        snare_weights=[1.0, 1.0],
        snare_velocity_range=(100, 115),
        use_clap=True,
        hihat_subdivision=8,
        hihat_open_positions=[0.5, 2.5],
        hihat_velocity_range=(60, 85),
        swing_amount=0.0,
    ),
    "house": RhythmGrammar(
        kick_positions=[0.0, 1.0, 2.0, 3.0],
        kick_weights=[1.0, 1.0, 1.0, 1.0],
        kick_velocity_range=(105, 125),
        snare_positions=[1.0, 3.0],
        snare_weights=[0.0, 0.0],  # No snare, clap instead
        use_clap=True,
        hihat_subdivision=8,
        hihat_open_positions=[0.5, 1.5, 2.5, 3.5],
        hihat_velocity_range=(55, 80),
        swing_amount=0.05,
    ),
    "techno": RhythmGrammar(
        kick_positions=[0.0, 1.0, 2.0, 3.0],
        kick_weights=[1.0, 1.0, 1.0, 1.0],
        kick_velocity_range=(110, 127),
        snare_positions=[1.0, 3.0],
        snare_weights=[0.5, 0.5],
        snare_velocity_range=(80, 100),
        use_clap=True,
        hihat_subdivision=16,
        hihat_open_positions=[],
        hihat_velocity_range=(50, 75),
        swing_amount=0.0,
    ),
    "drum-and-bass": RhythmGrammar(
        kick_positions=[0.0, 1.75, 2.5],
        kick_weights=[1.0, 0.6, 0.7],
        kick_velocity_range=(100, 120),
        snare_positions=[0.5, 1.0, 2.5, 3.0],
        snare_weights=[0.3, 1.0, 0.3, 1.0],
        snare_velocity_range=(95, 115),
        hihat_subdivision=16,
        hihat_velocity_range=(60, 90),
        hihat_variation=0.2,
        swing_amount=0.0,
        ghost_note_probability=0.1,
    ),
    "reggaeton": RhythmGrammar(
        kick_positions=[0.0, 0.75, 2.0, 2.75],
        kick_weights=[1.0, 0.8, 1.0, 0.8],
        kick_velocity_range=(100, 118),
        snare_positions=[0.5, 1.5, 2.5, 3.5],  # Dembow pattern
        snare_weights=[0.7, 1.0, 0.7, 1.0],
        snare_velocity_range=(90, 110),
        hihat_subdivision=8,
        hihat_velocity_range=(55, 80),
        swing_amount=0.0,
    ),
    "afrobeat": RhythmGrammar(
        kick_positions=[0.0, 1.5, 2.0, 3.25],
        kick_weights=[1.0, 0.6, 0.8, 0.5],
        kick_velocity_range=(90, 110),
        snare_positions=[1.0, 3.0],
        snare_weights=[1.0, 1.0],
        snare_velocity_range=(85, 105),
        hihat_subdivision=8,
        hihat_velocity_range=(50, 75),
        hihat_variation=0.15,
        swing_amount=0.15,
        ghost_note_probability=0.2,
    ),
    "pop": RhythmGrammar(
        kick_positions=[0.0, 2.0],
        kick_weights=[1.0, 1.0],
        kick_velocity_range=(100, 118),
        snare_positions=[1.0, 3.0],
        snare_weights=[1.0, 1.0],
        snare_velocity_range=(95, 115),
        hihat_subdivision=8,
        hihat_velocity_range=(55, 80),
        swing_amount=0.0,
    ),
    "cinematic": RhythmGrammar(
        kick_positions=[0.0, 2.0],
        kick_weights=[1.0, 0.5],
        kick_velocity_range=(80, 110),
        snare_positions=[3.0],
        snare_weights=[1.0],
        snare_velocity_range=(70, 100),
        hihat_subdivision=4,
        hihat_velocity_range=(40, 60),
        swing_amount=0.0,
        fill_probability=0.2,
    ),
}


class RhythmGenerator:
    """Generates rhythm patterns according to genre grammar."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_pattern(
        self,
        genre_id: str,
        variation: float = 0.0,
    ) -> RhythmPattern:
        """
        Generate a one-bar rhythm pattern for the genre.

        Args:
            genre_id: Target genre
            variation: Amount of variation from standard pattern (0-1)

        Returns:
            RhythmPattern with drum hits
        """
        grammar = GENRE_RHYTHM_GRAMMAR.get(genre_id)
        if grammar is None:
            grammar = GENRE_RHYTHM_GRAMMAR["pop"]

        hits = []

        # Generate kick hits
        hits.extend(self._generate_kicks(grammar, variation))

        # Generate snare/clap hits
        hits.extend(self._generate_snares(grammar, variation))

        # Generate hi-hat hits
        hits.extend(self._generate_hihats(grammar, variation))

        return RhythmPattern(
            hits=hits,
            swing_amount=grammar.swing_amount,
        )

    def _generate_kicks(
        self,
        grammar: RhythmGrammar,
        variation: float,
    ) -> list[DrumHit]:
        """Generate kick drum hits."""
        hits = []

        for pos, weight in zip(grammar.kick_positions, grammar.kick_weights):
            # Adjust probability with variation
            prob = weight * (1.0 - variation * 0.5)

            if self.rng.random() < prob:
                velocity = self.rng.randint(*grammar.kick_velocity_range)

                # Add slight timing variation
                timing_var = self.rng.uniform(-0.02, 0.02) * variation
                position = max(0, pos + timing_var)

                hits.append(DrumHit(
                    instrument="kick",
                    position=position,
                    velocity=velocity,
                ))

        return hits

    def _generate_snares(
        self,
        grammar: RhythmGrammar,
        variation: float,
    ) -> list[DrumHit]:
        """Generate snare/clap hits."""
        hits = []
        instrument = "clap" if grammar.use_clap else "snare"

        for pos, weight in zip(grammar.snare_positions, grammar.snare_weights):
            if weight == 0:
                continue

            prob = weight * (1.0 - variation * 0.3)

            if self.rng.random() < prob:
                velocity = self.rng.randint(*grammar.snare_velocity_range)

                hits.append(DrumHit(
                    instrument=instrument,
                    position=pos,
                    velocity=velocity,
                ))

                # Ghost notes
                if grammar.ghost_note_probability > 0:
                    if self.rng.random() < grammar.ghost_note_probability:
                        ghost_pos = pos - 0.25
                        if ghost_pos >= 0:
                            hits.append(DrumHit(
                                instrument="snare",
                                position=ghost_pos,
                                velocity=velocity // 2,
                            ))

        return hits

    def _generate_hihats(
        self,
        grammar: RhythmGrammar,
        variation: float,
    ) -> list[DrumHit]:
        """Generate hi-hat hits."""
        hits = []

        # Calculate positions based on subdivision
        step = 4.0 / grammar.hihat_subdivision
        positions = [i * step for i in range(grammar.hihat_subdivision)]

        for pos in positions:
            # Determine if open or closed
            is_open = pos in grammar.hihat_open_positions
            is_accent = pos in grammar.hihat_accent_positions

            # Skip some hits with variation
            if self.rng.random() < variation * 0.3:
                continue

            # Base velocity
            velocity = self.rng.randint(*grammar.hihat_velocity_range)

            # Apply variation
            velocity = int(velocity * (1.0 + self.rng.uniform(
                -grammar.hihat_variation,
                grammar.hihat_variation
            )))

            # Accent
            if is_accent:
                velocity = min(127, velocity + 15)

            velocity = max(1, min(127, velocity))

            instrument = "hihat_open" if is_open else "hihat_closed"

            hits.append(DrumHit(
                instrument=instrument,
                position=pos,
                velocity=velocity,
            ))

        return hits

    def generate_fill(
        self,
        genre_id: str,
        fill_length_beats: float = 1.0,
    ) -> list[DrumHit]:
        """Generate a drum fill."""
        hits = []

        # Snare/tom roll for fill
        num_hits = int(fill_length_beats * 4)
        step = fill_length_beats / num_hits

        instruments = ["snare", "tom_high", "tom_mid", "tom_low"]

        for i in range(num_hits):
            instrument = instruments[min(i // 2, len(instruments) - 1)]
            velocity = 80 + i * 5  # Crescendo

            hits.append(DrumHit(
                instrument=instrument,
                position=i * step,
                velocity=min(127, velocity),
            ))

        return hits


def generate_rhythm_pattern(
    genre_id: str,
    num_bars: int = 4,
    variation: float = 0.0,
    seed: Optional[int] = None,
) -> list[RhythmPattern]:
    """
    Generate multiple bars of rhythm pattern.

    Args:
        genre_id: Target genre
        num_bars: Number of bars to generate
        variation: Variation amount (0-1)
        seed: Random seed for reproducibility

    Returns:
        List of RhythmPatterns
    """
    generator = RhythmGenerator(seed)
    patterns = []

    for i in range(num_bars):
        # Increase variation slightly for later bars
        bar_variation = variation * (1.0 + i * 0.1)
        pattern = generator.generate_pattern(genre_id, bar_variation)
        patterns.append(pattern)

    return patterns


def get_rhythm_grammar(genre_id: str) -> Optional[RhythmGrammar]:
    """Get rhythm grammar for a genre."""
    return GENRE_RHYTHM_GRAMMAR.get(genre_id)
