# AETHER Voice Generation & Singing Module

## Technical Specification v1.0

---

## 1. Vocal Identity Blueprint

### 1.1 Singer Profile: AETHER Voice Unit "AVU-1"

**Vocal Classification:** Lyric Tenor / High Baritone Hybrid
**Effective Range:** G2 – C5 (comfortable), E2 – E5 (extended with controlled strain)
**Tessitura:** C3 – G4 (optimal power and clarity zone)

### 1.2 Timbre Characteristics

| Parameter | Value | Description |
|-----------|-------|-------------|
| Brightness | 0.62 | Slight presence peak at 3-4kHz, not harsh |
| Breathiness | 0.25 | Clean tone with subtle air on soft passages |
| Grit/Rasp | 0.15 | Available on demand, not default |
| Nasality | 0.18 | Minimal, controlled for genre needs |
| Chest Resonance | 0.70 | Full lower-mid body |
| Head Voice Blend | 0.55 | Smooth transition, no obvious break |

### 1.3 Formant Profile (Hz)

```
F1 (openness):    500-700 Hz baseline
F2 (frontness):   1400-1800 Hz baseline
F3 (brightness):  2400-2800 Hz baseline
Singer's Formant: 2800-3200 Hz (present but not exaggerated)
```

### 1.4 Emotional Baseline

- **Default State:** Warm, controlled intimacy with latent power
- **Tonal Center:** Conversational sincerity, not theatrical
- **Energy Floor:** Never sounds bored or disengaged
- **Energy Ceiling:** Powerful without screaming or distortion

### 1.5 Identity Invariants ("Do Not Drift" List)

These traits remain constant across ALL genres and emotional states:

| Invariant | Specification |
|-----------|---------------|
| Fundamental Timbre | Core formant relationships preserved within ±5% |
| Vowel Signature | Characteristic 'a' and 'o' shaping maintained |
| Vibrato DNA | Rate 5.2-5.8 Hz, onset delay 180-280ms |
| Breath Sound | Consistent inhale/exhale character |
| Transition Smoothness | No hard register breaks |
| Consonant Articulation | Consistent attack/release profile |
| Sibilance Character | Controlled 's' and 'sh' brightness |

### 1.6 Controlled Flexibility List

These traits adapt by genre while preserving identity:

| Flexible Trait | Range | Genre Influence |
|----------------|-------|-----------------|
| Vibrato Depth | 0.1 – 0.8 semitones | Pop: moderate, R&B: wide, Rock: tight |
| Breathiness | 0.1 – 0.5 | Ambient: high, Rock: low |
| Grit Activation | 0.0 – 0.6 | Rock/Blues: high, Pop: low |
| Dynamics Range | 6 – 18 dB | Jazz: wide, EDM: compressed |
| Rhythmic Pocket | -30ms to +20ms | Trap: behind, Funk: on-top |
| Register Balance | 30/70 – 70/30 chest/head | Genre-dependent |
| Ornamentation Density | 0.0 – 0.7 | R&B: high, Rock: low |

---

## 2. Language & Phonetics Architecture

### 2.1 Phoneme Inventory

#### English (General American Neutral)

```
Vowels (15):
  Monophthongs: /i/, /ɪ/, /e/, /ɛ/, /æ/, /ɑ/, /ɔ/, /o/, /ʊ/, /u/, /ʌ/, /ə/
  Diphthongs: /aɪ/, /aʊ/, /ɔɪ/

Consonants (24):
  Stops: /p/, /b/, /t/, /d/, /k/, /g/
  Fricatives: /f/, /v/, /θ/, /ð/, /s/, /z/, /ʃ/, /ʒ/, /h/
  Affricates: /tʃ/, /dʒ/
  Nasals: /m/, /n/, /ŋ/
  Liquids: /l/, /r/
  Glides: /w/, /j/
```

#### Spanish (Neutral Latin American)

```
Vowels (5):
  /a/, /e/, /i/, /o/, /u/

Consonants (19):
  Stops: /p/, /b/, /t/, /d/, /k/, /g/
  Fricatives: /f/, /s/, /x/, /β/, /ð/, /ɣ/
  Affricates: /tʃ/
  Nasals: /m/, /n/, /ɲ/
  Liquids: /l/, /r/, /ɾ/
  Glides: /w/, /j/
```

### 2.2 Prosody Systems

#### English Prosody Rules

```python
class EnglishProsody:
    stress_timing = True  # Stress-timed language
    reduction_enabled = True  # Unstressed vowels reduce to schwa

    linking_rules = {
        "consonant_to_vowel": True,   # "run_away" → "ru-na-way"
        "r_linking": True,            # "far_away" → "fa-ra-way"
        "glottal_insertion": False,   # Avoid for singing
    }

    stress_patterns = {
        "default_noun": "initial",
        "default_verb": "final",
        "compound": "first_element",
    }
```

#### Spanish Prosody Rules

```python
class SpanishProsody:
    stress_timing = False  # Syllable-timed language
    reduction_enabled = False  # All vowels maintain full quality

    stress_rules = {
        "ends_vowel_n_s": "penultimate",
        "ends_consonant": "ultimate",
        "accent_mark": "override",
    }

    synalepha = True  # Merge vowels across word boundaries
```

### 2.3 Singing-Specific Vowel Shaping

| Vowel | English Target | Spanish Target | Singing Modification |
|-------|----------------|----------------|---------------------|
| /a/ | Open, back | Open, central | Raise soft palate, maximize resonance |
| /i/ | Close, front | Close, front | Slight opening above D4 to avoid tension |
| /u/ | Close, back | Close, back | Forward placement to maintain clarity |
| /e/ | Mid, front | Mid, front | Stable jaw, tongue forward |
| /o/ | Mid, back | Mid, back | Rounded lips, lowered larynx |

### 2.4 Accent Neutrality Strategy

**English Target:** Mid-Atlantic Neutral
- No regional markers (no Boston 'r', no Southern drawl)
- Clear 't' sounds (not glottalized)
- Full 'r' coloring (rhotic)

**Spanish Target:** Neutral Latin American
- Seseo (no distinction between 's' and 'z')
- No aspiration of 's'
- Clear 'll' as /j/ (yeísmo)
- No regional vocabulary markers

### 2.5 Bilingual Handling (Code-Switching)

```python
class BilingualController:
    def handle_language_switch(self, current_lang: str, next_lang: str,
                                boundary_type: str) -> TransitionParams:
        """
        Manage smooth transitions between languages within songs.
        """
        if boundary_type == "phrase_break":
            # Full language reset at phrase boundaries
            return TransitionParams(
                blend_frames=0,
                reset_prosody=True,
                maintain_timbre=True
            )
        elif boundary_type == "word_boundary":
            # Smooth blend for mid-phrase switches
            return TransitionParams(
                blend_frames=3,
                reset_prosody=False,
                maintain_timbre=True,
                interpolate_formants=True
            )
        elif boundary_type == "loan_word":
            # Preserve source language pronunciation
            return TransitionParams(
                use_source_phonemes=True,
                adapt_to_target_prosody=True
            )
```

### 2.6 Acceptance Criteria

#### Intelligibility

| Metric | Threshold | Measurement |
|--------|-----------|-------------|
| Word Recognition Rate | ≥ 95% | Native listener transcription test |
| Phoneme Accuracy | ≥ 98% | Forced alignment scoring |
| Cross-Language Confusion | ≤ 2% | Bilingual listener identification |

#### Natural Stress

| Metric | Threshold | Measurement |
|--------|-----------|-------------|
| Stress Placement Accuracy | ≥ 97% | Linguistic analysis |
| Prosodic Naturalness (MOS) | ≥ 4.2/5.0 | Human evaluation |
| Duration Ratio Correctness | Within 15% of reference | Acoustic analysis |

#### No Phoneme Bleeding

| Metric | Threshold | Measurement |
|--------|-----------|-------------|
| Language Contamination Rate | ≤ 1% | Expert phonetic review |
| Accent Consistency | ≥ 95% within language | Listener panel |
| Formant Stability | Within ±8% of target | Spectrogram analysis |

---

## 3. Singing Engine Design

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    AETHER Singing Engine                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │   Lyric   │───▶│   Phoneme    │───▶│   Duration        │   │
│  │   Input   │    │   Converter  │    │   Predictor       │   │
│  └───────────┘    └──────────────┘    └─────────┬─────────┘   │
│                                                  │              │
│  ┌───────────┐    ┌──────────────┐    ┌─────────▼─────────┐   │
│  │  Melody   │───▶│   Alignment  │───▶│   Pitch           │   │
│  │   Input   │    │   Engine     │    │   Contour Gen     │   │
│  └───────────┘    └──────────────┘    └─────────┬─────────┘   │
│                                                  │              │
│  ┌───────────┐    ┌──────────────┐    ┌─────────▼─────────┐   │
│  │  Genre    │───▶│   Style      │───▶│   Expression      │   │
│  │  Context  │    │   Encoder    │    │   Modulator       │   │
│  └───────────┘    └──────────────┘    └─────────┬─────────┘   │
│                                                  │              │
│                              ┌───────────────────▼───────────┐ │
│                              │      Neural Vocoder          │ │
│                              │   (Acoustic Model + Decoder) │ │
│                              └───────────────────┬───────────┘ │
│                                                  │              │
│                              ┌───────────────────▼───────────┐ │
│                              │       Audio Output           │ │
│                              │    (48kHz, 24-bit float)     │ │
│                              └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Input Specification

```python
@dataclass
class SingingEngineInput:
    # Required
    lyrics: List[LyricToken]           # Tokenized lyrics with timing hints
    melody: List[MelodyNote]           # MIDI-like note sequence
    tempo: float                        # BPM
    key: Key                           # Musical key
    time_signature: TimeSignature

    # Required Context
    genre_id: str                      # Genre DNA identifier
    language: Literal["en", "es"]      # Primary language

    # Optional Expression
    emotion_curve: Optional[List[EmotionPoint]] = None
    energy_curve: Optional[List[float]] = None
    section_map: Optional[List[SectionMarker]] = None

    # Optional Control
    style_overrides: Optional[VocalStyleParams] = None

@dataclass
class LyricToken:
    text: str                          # Word or syllable
    phonemes: Optional[List[str]]      # Override phonemes
    language: Literal["en", "es"]      # Per-token language
    stress: Optional[float]            # 0.0-1.0 stress weight

@dataclass
class MelodyNote:
    pitch: int                         # MIDI note number
    start_beat: float                  # Position in beats
    duration_beats: float              # Length in beats
    velocity: int                      # 0-127 intensity hint
    lyric_index: Optional[int]         # Link to lyric token
```

### 3.3 Output Specification

```python
@dataclass
class SingingEngineOutput:
    audio: np.ndarray                  # Shape: (samples,), float32, 48kHz
    sample_rate: int = 48000

    # Alignment data for downstream processing
    phoneme_alignment: List[PhonemeSpan]
    word_alignment: List[WordSpan]

    # Quality metrics
    pitch_confidence: List[float]      # Per-frame confidence
    phoneme_confidence: List[float]    # Per-phoneme confidence

    # Metadata
    duration_seconds: float
    render_params: VocalRenderParams   # Parameters used
```

### 3.4 Lyric-to-Note Alignment Logic

```python
class LyricMelodyAligner:
    """
    Aligns lyric syllables to melody notes with singing-aware heuristics.
    """

    def align(self, lyrics: List[LyricToken],
              melody: List[MelodyNote]) -> List[AlignedUnit]:

        # Step 1: Syllabification
        syllables = self._syllabify(lyrics)

        # Step 2: Initial assignment (one syllable per note)
        assignments = self._initial_assignment(syllables, melody)

        # Step 3: Melisma detection (multiple notes per syllable)
        assignments = self._detect_melismas(assignments, melody)

        # Step 4: Duration allocation within syllables
        assignments = self._allocate_phoneme_durations(assignments)

        # Step 5: Transition planning
        assignments = self._plan_transitions(assignments)

        return assignments

    def _detect_melismas(self, assignments: List,
                         melody: List[MelodyNote]) -> List:
        """
        Identify where single syllables span multiple notes.

        Heuristics:
        - Consecutive notes without new syllable onset
        - Vowel-heavy syllables can sustain longer
        - Genre influences melisma likelihood
        """
        melisma_threshold = self.genre_params.melisma_density

        for i, note in enumerate(melody):
            if note.lyric_index is None:
                # Unassigned note - check for melisma continuation
                prev_syllable = self._get_prev_syllable(assignments, i)
                if self._can_sustain(prev_syllable, note, melisma_threshold):
                    assignments[i].extend_from = prev_syllable

        return assignments
```

### 3.5 Pitch Accuracy vs Expressive Deviation

```python
class PitchController:
    """
    Balances pitch accuracy with musical expression.
    """

    # Deviation budget by context
    DEVIATION_RULES = {
        "note_attack": {
            "scoop_up": (-50, 0),      # cents below target
            "scoop_down": (0, 30),     # cents above target
            "duration_ms": (20, 80),   # transition time
        },
        "note_sustain": {
            "drift_range": (-10, 10),  # cents
            "vibrato_center": 0,       # target pitch
        },
        "note_release": {
            "fall_range": (0, -100),   # cents below
            "rise_range": (0, 50),     # cents above
        },
    }

    def generate_pitch_contour(self, note: MelodyNote,
                               context: PitchContext) -> np.ndarray:
        """
        Generate frame-by-frame pitch contour for a note.
        """
        frames = self._note_to_frames(note)
        contour = np.zeros(frames)

        # Base pitch
        target_hz = midi_to_hz(note.pitch)
        contour[:] = target_hz

        # Attack deviation (genre-dependent)
        attack_type = self._select_attack(context)
        contour = self._apply_attack(contour, attack_type)

        # Sustain vibrato
        vibrato = self._generate_vibrato(note, context)
        contour = self._apply_vibrato(contour, vibrato)

        # Release behavior
        release_type = self._select_release(context)
        contour = self._apply_release(contour, release_type)

        return contour
```

### 3.6 Vibrato Modeling

```python
@dataclass
class VibratoParams:
    rate_hz: float = 5.5              # Oscillation frequency
    depth_cents: float = 40           # Pitch deviation
    onset_delay_ms: float = 200       # Time before vibrato starts
    attack_ms: float = 150            # Ramp-up time
    irregularity: float = 0.15        # Humanization factor

class VibratoGenerator:
    """
    Generates natural, genre-appropriate vibrato.
    """

    GENRE_PRESETS = {
        "pop": VibratoParams(rate_hz=5.5, depth_cents=35, onset_delay_ms=180),
        "r-and-b": VibratoParams(rate_hz=5.0, depth_cents=60, onset_delay_ms=150),
        "rock": VibratoParams(rate_hz=6.0, depth_cents=25, onset_delay_ms=250),
        "jazz": VibratoParams(rate_hz=5.2, depth_cents=50, onset_delay_ms=200),
        "classical": VibratoParams(rate_hz=5.8, depth_cents=45, onset_delay_ms=100),
    }

    def generate(self, duration_frames: int, params: VibratoParams,
                 note_velocity: int) -> np.ndarray:
        """
        Generate vibrato modulation signal.
        """
        # Base sinusoid with slight irregularity
        t = np.arange(duration_frames) / self.frame_rate
        phase_noise = self._generate_phase_noise(duration_frames, params.irregularity)

        vibrato = params.depth_cents * np.sin(
            2 * np.pi * params.rate_hz * t + phase_noise
        )

        # Onset envelope
        onset_frames = int(params.onset_delay_ms * self.frame_rate / 1000)
        attack_frames = int(params.attack_ms * self.frame_rate / 1000)

        envelope = np.ones(duration_frames)
        envelope[:onset_frames] = 0
        envelope[onset_frames:onset_frames + attack_frames] = np.linspace(
            0, 1, attack_frames
        )

        # Velocity scaling (softer notes = less vibrato)
        velocity_scale = 0.5 + 0.5 * (note_velocity / 127)

        return vibrato * envelope * velocity_scale
```

### 3.7 Transition Types

```python
class TransitionEngine:
    """
    Models transitions between notes with singing-appropriate behaviors.
    """

    TRANSITION_TYPES = {
        "legato": {
            "description": "Smooth, connected transition",
            "pitch_behavior": "glide",
            "glide_duration_ratio": 0.15,  # % of note duration
            "breath_break": False,
        },
        "portamento": {
            "description": "Deliberate pitch slide",
            "pitch_behavior": "slide",
            "glide_duration_ratio": 0.25,
            "breath_break": False,
        },
        "staccato": {
            "description": "Separated, punctuated",
            "pitch_behavior": "step",
            "note_shortening": 0.3,
            "breath_break": False,
        },
        "breath": {
            "description": "Breath-separated phrases",
            "pitch_behavior": "step",
            "breath_duration_ms": (150, 400),
            "breath_break": True,
        },
        "scoop": {
            "description": "Approach from below",
            "pitch_behavior": "scoop_up",
            "scoop_cents": (-100, -30),
            "scoop_duration_ms": (30, 80),
        },
        "fall": {
            "description": "Release downward",
            "pitch_behavior": "fall_off",
            "fall_cents": (-200, -50),
            "fall_duration_ms": (50, 150),
        },
    }

    def select_transition(self, prev_note: MelodyNote, next_note: MelodyNote,
                          context: TransitionContext) -> str:
        """
        Select appropriate transition based on musical context.
        """
        interval = next_note.pitch - prev_note.pitch
        time_gap = next_note.start_beat - (prev_note.start_beat + prev_note.duration_beats)

        # Phrase boundary detection
        if time_gap > context.phrase_break_threshold:
            return "breath"

        # Large intervals often get portamento
        if abs(interval) > 4 and context.genre_allows_portamento:
            return "portamento"

        # Genre-specific defaults
        return context.genre_default_transition
```

### 3.8 Breath Modeling

```python
class BreathModel:
    """
    Models realistic breath sounds and phrasing.
    """

    def plan_breaths(self, lyrics: List[LyricToken],
                     melody: List[MelodyNote],
                     section_map: List[SectionMarker]) -> List[BreathEvent]:
        """
        Determine breath placement based on:
        1. Phrase structure (punctuation, line breaks)
        2. Available gaps between notes
        3. Physiological constraints (can't sing forever)
        4. Musical phrasing
        """
        breaths = []
        accumulated_duration = 0.0
        max_phrase_duration = 8.0  # seconds before forced breath

        for i, note in enumerate(melody):
            accumulated_duration += note.duration_beats / self.tempo * 60

            # Check for natural breath points
            is_phrase_end = self._is_phrase_boundary(lyrics, i)
            has_gap = self._has_sufficient_gap(melody, i)
            needs_breath = accumulated_duration > max_phrase_duration

            if (is_phrase_end or needs_breath) and has_gap:
                breath_type = self._select_breath_type(
                    accumulated_duration,
                    self._get_next_phrase_intensity(section_map, i)
                )
                breaths.append(BreathEvent(
                    position=note.start_beat + note.duration_beats,
                    duration_ms=breath_type.duration,
                    intensity=breath_type.intensity,
                    audible=breath_type.audible
                ))
                accumulated_duration = 0.0

        return breaths

    def synthesize_breath(self, event: BreathEvent) -> np.ndarray:
        """
        Generate breath audio sample.
        """
        # Breath is combination of:
        # - Filtered noise (inhale character)
        # - Formant shaping (vocalist identity)
        # - Amplitude envelope (natural dynamics)

        duration_samples = int(event.duration_ms * self.sample_rate / 1000)

        # Pink noise base
        noise = self._generate_pink_noise(duration_samples)

        # Shape with vocalist's formants (reduced)
        breath_formants = self.vocalist_formants * 0.3
        shaped = self._apply_formants(noise, breath_formants)

        # Envelope
        envelope = self._breath_envelope(duration_samples, event.intensity)

        return shaped * envelope * event.intensity
```

### 3.9 Differences from Standard TTS

| Aspect | TTS | Singing Engine |
|--------|-----|----------------|
| Pitch Source | Prosody model prediction | External melody input |
| Duration | Model-predicted, speech-natural | Note-aligned, music-synchronized |
| Phoneme Timing | Variable, context-dependent | Beat-locked with expression |
| Pitch Range | ~1 octave typical | 2-3 octaves required |
| Vibrato | None/minimal | Essential, stylized |
| Breath | Natural speech pacing | Musical phrase structure |
| Dynamics | Limited range | Wide range, genre-dependent |
| Tempo | Speech rate | Fixed BPM, synced |

### 3.10 Integration with Melody Generation

```python
class VocalIntegrationLayer:
    """
    Connects singing engine to AETHER melody generation pipeline.
    """

    async def process_vocal_request(self,
                                    melody_spec: MelodySpec,
                                    lyrics: LyricsSpec,
                                    genre_dna: GenreDNA) -> VocalOutput:
        """
        Full pipeline from specs to rendered vocals.
        """
        # 1. Extract singable melody from spec
        vocal_melody = self._extract_vocal_line(melody_spec)

        # 2. Process lyrics
        processed_lyrics = await self.lyric_processor.process(
            lyrics.text,
            lyrics.language,
            melody_spec.time_signature
        )

        # 3. Align lyrics to melody
        aligned = self.aligner.align(processed_lyrics, vocal_melody)

        # 4. Apply genre styling
        styled = self.style_encoder.apply(aligned, genre_dna)

        # 5. Generate expression curves
        expression = self.expression_generator.generate(
            styled,
            melody_spec.section_map,
            genre_dna.vocal_style
        )

        # 6. Render vocals
        vocals = await self.singing_engine.render(
            styled,
            expression,
            self.vocalist_identity
        )

        # 7. Post-process
        processed = self.vocal_processor.process(
            vocals,
            genre_dna.vocal_processing
        )

        return VocalOutput(
            audio=processed,
            alignment=aligned,
            metadata=self._build_metadata(styled, expression)
        )
```

---

## 4. Genre-Aware Vocal Performance Layer

### 4.1 Performance Parameter Matrix

| Genre | Rhythmic Pocket | Articulation | Intensity | Ornamentation | Register |
|-------|-----------------|--------------|-----------|---------------|----------|
| Pop | On-grid (±5ms) | Legato-medium | 0.6-0.8 | Low (0.2) | 60/40 chest |
| R&B | Behind (-15ms) | Legato-heavy | 0.5-0.9 | High (0.7) | 50/50 |
| Rock | On-grid to ahead | Staccato-medium | 0.7-1.0 | Low (0.15) | 80/20 chest |
| Jazz | Behind (-20ms) | Legato | 0.4-0.8 | Medium (0.4) | 40/60 head |
| House | On-grid (±3ms) | Staccato | 0.5-0.7 | None (0.05) | 55/45 |
| Trap | Behind (-25ms) | Staccato-heavy | 0.4-0.7 | Medium (0.35) | 70/30 chest |
| Funk | Ahead (+10ms) | Staccato-punchy | 0.7-0.9 | Medium (0.3) | 75/25 chest |
| Ambient | Free (±30ms) | Extreme legato | 0.2-0.5 | Low (0.1) | 30/70 head |
| Latin | On-grid | Medium | 0.6-0.85 | Medium (0.4) | 55/45 |

### 4.2 Genre Performance Profiles

```python
@dataclass
class VocalPerformanceProfile:
    genre_id: str

    # Timing
    rhythmic_offset_ms: Tuple[float, float]  # (min, max) from grid
    swing_amount: float                       # 0.0-1.0
    anticipation_tendency: float              # -1.0 (behind) to 1.0 (ahead)

    # Articulation
    default_transition: str                   # legato, staccato, etc.
    note_attack_sharpness: float             # 0.0-1.0
    note_release_type: str                   # sustain, cut, fall
    consonant_emphasis: float                # 0.0-1.0

    # Expression
    dynamic_range_db: Tuple[float, float]
    intensity_baseline: float
    emotion_responsiveness: float            # How much emotion affects performance

    # Ornamentation
    run_probability: float                   # Chance of melodic runs
    bend_probability: float                  # Chance of pitch bends
    ad_lib_density: float                    # Frequency of improvised elements
    vibrato_depth_scale: float               # Multiplier on base vibrato

    # Register
    chest_voice_preference: float            # 0.0-1.0
    falsetto_threshold: int                  # MIDI note to switch
    mix_voice_range: Tuple[int, int]        # Notes for blended register


GENRE_PROFILES = {
    "pop": VocalPerformanceProfile(
        genre_id="pop",
        rhythmic_offset_ms=(-5, 5),
        swing_amount=0.0,
        anticipation_tendency=0.0,
        default_transition="legato",
        note_attack_sharpness=0.5,
        note_release_type="sustain",
        consonant_emphasis=0.6,
        dynamic_range_db=(6, 12),
        intensity_baseline=0.7,
        emotion_responsiveness=0.6,
        run_probability=0.1,
        bend_probability=0.15,
        ad_lib_density=0.05,
        vibrato_depth_scale=0.8,
        chest_voice_preference=0.6,
        falsetto_threshold=72,  # C5
        mix_voice_range=(64, 72),
    ),

    "r-and-b": VocalPerformanceProfile(
        genre_id="r-and-b",
        rhythmic_offset_ms=(-20, -5),
        swing_amount=0.15,
        anticipation_tendency=-0.3,
        default_transition="legato",
        note_attack_sharpness=0.3,
        note_release_type="fall",
        consonant_emphasis=0.4,
        dynamic_range_db=(8, 18),
        intensity_baseline=0.6,
        emotion_responsiveness=0.9,
        run_probability=0.4,
        bend_probability=0.5,
        ad_lib_density=0.25,
        vibrato_depth_scale=1.3,
        chest_voice_preference=0.5,
        falsetto_threshold=67,  # G4
        mix_voice_range=(60, 70),
    ),

    "rock": VocalPerformanceProfile(
        genre_id="rock",
        rhythmic_offset_ms=(-5, 10),
        swing_amount=0.0,
        anticipation_tendency=0.2,
        default_transition="staccato",
        note_attack_sharpness=0.8,
        note_release_type="cut",
        consonant_emphasis=0.85,
        dynamic_range_db=(6, 15),
        intensity_baseline=0.8,
        emotion_responsiveness=0.7,
        run_probability=0.02,
        bend_probability=0.2,
        ad_lib_density=0.08,
        vibrato_depth_scale=0.6,
        chest_voice_preference=0.85,
        falsetto_threshold=76,  # E5
        mix_voice_range=(67, 74),
    ),

    # Additional genres defined similarly...
}
```

### 4.3 Ornamentation Engine

```python
class OrnamentationEngine:
    """
    Adds genre-appropriate vocal ornaments while preserving identity.
    """

    ORNAMENT_TYPES = {
        "run": {
            "description": "Fast scalar passage",
            "note_count": (3, 8),
            "duration_ratio": 0.3,  # Of original note
            "genres": ["r-and-b", "gospel", "jazz"],
        },
        "turn": {
            "description": "Upper/lower neighbor figure",
            "note_count": 4,
            "duration_ratio": 0.15,
            "genres": ["pop", "r-and-b", "latin"],
        },
        "mordent": {
            "description": "Quick alternation",
            "note_count": 3,
            "duration_ratio": 0.08,
            "genres": ["jazz", "classical"],
        },
        "scoop": {
            "description": "Approach from below",
            "pitch_delta": (-100, -30),  # cents
            "genres": ["pop", "rock", "country"],
        },
        "fall": {
            "description": "Release downward",
            "pitch_delta": (-200, -50),
            "genres": ["pop", "r-and-b", "hip-hop"],
        },
        "flip": {
            "description": "Quick octave jump",
            "genres": ["r-and-b", "pop"],
        },
    }

    def apply_ornaments(self, melody: List[AlignedUnit],
                        profile: VocalPerformanceProfile) -> List[AlignedUnit]:
        """
        Add ornaments based on genre profile and musical context.
        """
        ornamented = []

        for i, unit in enumerate(melody):
            # Determine ornament probability for this note
            can_ornament = self._can_ornament(unit, melody, i)

            if can_ornament and random.random() < profile.run_probability:
                ornament_type = self._select_ornament(unit, profile)
                unit = self._apply_ornament(unit, ornament_type, profile)

            # Always consider pitch bends (more subtle)
            if random.random() < profile.bend_probability:
                unit = self._apply_bend(unit, profile)

            ornamented.append(unit)

        return ornamented

    def _select_ornament(self, unit: AlignedUnit,
                         profile: VocalPerformanceProfile) -> str:
        """
        Select contextually appropriate ornament.
        """
        # End of phrase favors falls
        if unit.is_phrase_end:
            return "fall"

        # Start of phrase favors scoops
        if unit.is_phrase_start:
            return "scoop"

        # Long notes can have runs
        if unit.duration_beats > 1.0:
            return random.choice(["run", "turn"])

        return "turn"
```

---

## 5. Vocal Arrangement System

### 5.1 Layer Architecture

```python
class VocalArrangementSystem:
    """
    Manages multi-layer vocal arrangements.
    """

    LAYER_TYPES = {
        "lead": {
            "description": "Primary melodic voice",
            "count": 1,
            "pan": 0.0,
            "level_db": 0.0,
            "processing": "lead_chain",
        },
        "double": {
            "description": "Tight unison reinforcement",
            "count": (1, 2),
            "pan": (-0.15, 0.15),
            "level_db": -6.0,
            "timing_offset_ms": (-8, 8),
            "pitch_offset_cents": (-5, 5),
            "processing": "double_chain",
        },
        "harmony_high": {
            "description": "Upper harmony (3rd, 5th)",
            "count": (1, 2),
            "pan": (-0.4, -0.2),
            "level_db": -4.0,
            "interval": ["3rd", "5th"],
            "processing": "harmony_chain",
        },
        "harmony_low": {
            "description": "Lower harmony (3rd below, 5th below)",
            "count": (1, 2),
            "pan": (0.2, 0.4),
            "level_db": -4.0,
            "interval": ["-3rd", "-5th"],
            "processing": "harmony_chain",
        },
        "octave_up": {
            "description": "Octave doubling above",
            "count": 1,
            "pan": 0.0,
            "level_db": -8.0,
            "interval": "8va",
            "processing": "octave_chain",
        },
        "octave_down": {
            "description": "Octave doubling below",
            "count": 1,
            "pan": 0.0,
            "level_db": -10.0,
            "interval": "8vb",
            "processing": "octave_chain",
        },
        "whisper": {
            "description": "Breathy background texture",
            "count": (2, 4),
            "pan": (-0.8, 0.8),
            "level_db": -12.0,
            "breathiness_boost": 0.5,
            "processing": "whisper_chain",
        },
        "stack": {
            "description": "Background vocal stack",
            "count": (4, 8),
            "pan": "spread",
            "level_db": -8.0,
            "processing": "stack_chain",
        },
    }
```

### 5.2 Harmony Generation Rules

```python
class HarmonyGenerator:
    """
    Generates harmonies that respect music theory and vocalist range.
    """

    def generate_harmony(self, lead_melody: List[MelodyNote],
                         harmony_type: str,
                         key: Key,
                         voice_range: VocalRange) -> List[MelodyNote]:
        """
        Generate harmony line following voice leading principles.
        """
        harmony = []

        for i, note in enumerate(lead_melody):
            # Determine scale degree
            degree = self._get_scale_degree(note.pitch, key)

            # Select interval based on harmony type and chord context
            interval = self._select_interval(
                degree,
                harmony_type,
                self._get_chord_context(lead_melody, i)
            )

            # Calculate harmony pitch
            harmony_pitch = self._calculate_harmony_pitch(
                note.pitch,
                interval,
                key
            )

            # Ensure within singer's range
            harmony_pitch = self._constrain_to_range(harmony_pitch, voice_range)

            # Apply voice leading (minimize jumps)
            if i > 0:
                harmony_pitch = self._apply_voice_leading(
                    harmony_pitch,
                    harmony[-1].pitch,
                    voice_range
                )

            harmony.append(MelodyNote(
                pitch=harmony_pitch,
                start_beat=note.start_beat,
                duration_beats=note.duration_beats,
                velocity=int(note.velocity * 0.85),  # Slightly softer
                lyric_index=note.lyric_index
            ))

        return harmony

    HARMONY_INTERVALS = {
        "3rd": {
            "major_scale": [4, 3, 4, 3, 4, 3, 3],  # Semitones above by degree
            "minor_scale": [3, 4, 3, 4, 3, 4, 4],
        },
        "5th": {
            "major_scale": [7, 7, 7, 7, 7, 7, 6],
            "minor_scale": [7, 7, 7, 7, 6, 7, 7],
        },
        "-3rd": {
            "major_scale": [-3, -4, -3, -4, -3, -4, -4],
            "minor_scale": [-4, -3, -4, -3, -4, -3, -3],
        },
    }
```

### 5.3 Genre-Dependent Vocal Density

```python
GENRE_VOCAL_DENSITY = {
    "pop": {
        "verse": ["lead", "double"],
        "pre_chorus": ["lead", "double", "harmony_high"],
        "chorus": ["lead", "double", "harmony_high", "harmony_low", "stack"],
        "bridge": ["lead", "whisper"],
    },
    "r-and-b": {
        "verse": ["lead"],
        "pre_chorus": ["lead", "harmony_high"],
        "chorus": ["lead", "harmony_high", "harmony_low", "double"],
        "bridge": ["lead", "whisper", "harmony_high"],
        "ad_libs": ["lead"],  # Improvised fills
    },
    "rock": {
        "verse": ["lead"],
        "pre_chorus": ["lead", "double"],
        "chorus": ["lead", "double", "octave_up", "stack"],
        "bridge": ["lead"],
    },
    "ambient": {
        "all": ["lead", "whisper", "octave_up"],
    },
    "trap": {
        "verse": ["lead", "double"],
        "hook": ["lead", "double", "harmony_high", "octave_down"],
    },
}
```

### 5.4 Phase and Clutter Safeguards

```python
class VocalMixSafeguards:
    """
    Prevents phase issues and frequency masking in vocal arrangements.
    """

    def validate_arrangement(self, layers: List[VocalLayer]) -> ValidationResult:
        """
        Check for potential mix issues.
        """
        issues = []

        # Check for phase correlation issues
        for i, layer_a in enumerate(layers):
            for layer_b in layers[i+1:]:
                correlation = self._compute_phase_correlation(layer_a, layer_b)
                if correlation < -0.3:
                    issues.append(PhaseIssue(
                        layer_a=layer_a.name,
                        layer_b=layer_b.name,
                        correlation=correlation,
                        fix="Apply timing offset or pan separation"
                    ))

        # Check for frequency masking
        freq_analysis = self._analyze_frequency_overlap(layers)
        for overlap in freq_analysis.critical_overlaps:
            issues.append(MaskingIssue(
                layers=overlap.layers,
                frequency_range=overlap.range,
                fix="EQ separation or level adjustment"
            ))

        # Check total layer count
        if len(layers) > 12:
            issues.append(DensityIssue(
                count=len(layers),
                max_recommended=12,
                fix="Reduce layer count or combine similar parts"
            ))

        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            recommendations=self._generate_recommendations(issues)
        )

    def auto_fix(self, layers: List[VocalLayer],
                 issues: List[MixIssue]) -> List[VocalLayer]:
        """
        Automatically correct common issues.
        """
        for issue in issues:
            if isinstance(issue, PhaseIssue):
                # Add small timing offset to one layer
                layers = self._add_timing_offset(
                    layers,
                    issue.layer_b,
                    random.uniform(5, 15)  # ms
                )
            elif isinstance(issue, MaskingIssue):
                # Apply complementary EQ
                layers = self._apply_eq_separation(
                    layers,
                    issue.layers,
                    issue.frequency_range
                )

        return layers
```

---

## 6. Quality Control & Evaluation Framework

### 6.1 Automated Metrics

```python
class VocalQualityAnalyzer:
    """
    Automated quality analysis for synthesized vocals.
    """

    def analyze(self, vocals: np.ndarray,
                reference_melody: List[MelodyNote],
                lyrics: List[LyricToken]) -> QualityReport:
        """
        Comprehensive quality analysis.
        """
        return QualityReport(
            pitch_metrics=self._analyze_pitch(vocals, reference_melody),
            timing_metrics=self._analyze_timing(vocals, reference_melody),
            phoneme_metrics=self._analyze_phonemes(vocals, lyrics),
            timbre_metrics=self._analyze_timbre(vocals),
            overall_score=self._calculate_overall_score()
        )
```

### 6.2 Pitch Stability Metrics

| Metric | Calculation | Pass Threshold |
|--------|-------------|----------------|
| Pitch Accuracy | Mean absolute error vs target (cents) | ≤ 25 cents |
| Pitch Stability | Std dev within sustained notes (cents) | ≤ 15 cents |
| Vibrato Regularity | Std dev of vibrato rate (Hz) | ≤ 0.4 Hz |
| Intonation Score | % of frames within ±50 cents | ≥ 95% |
| Interval Accuracy | Error in melodic intervals (cents) | ≤ 20 cents |

```python
def analyze_pitch(self, vocals: np.ndarray,
                  melody: List[MelodyNote]) -> PitchMetrics:
    # Extract pitch contour
    f0, voiced_flag, _ = librosa.pyin(
        vocals,
        fmin=80,
        fmax=800,
        sr=self.sample_rate
    )

    # Align to reference
    aligned_ref = self._align_reference(melody, len(f0))

    # Calculate metrics
    pitch_error = np.abs(hz_to_cents(f0, aligned_ref))

    return PitchMetrics(
        accuracy_cents=np.nanmean(pitch_error),
        stability_cents=np.nanstd(pitch_error),
        intonation_score=np.mean(pitch_error < 50),
        max_deviation_cents=np.nanmax(pitch_error),
    )
```

### 6.3 Timing Alignment Scores

| Metric | Calculation | Pass Threshold |
|--------|-------------|----------------|
| Onset Accuracy | Mean onset error (ms) | ≤ 30 ms |
| Duration Accuracy | Mean duration error (%) | ≤ 10% |
| Syllable Alignment | % syllables correctly timed | ≥ 95% |
| Phrase Sync | Correlation with beat grid | ≥ 0.92 |

### 6.4 Phoneme Clarity Scoring

```python
class PhonemeEvaluator:
    """
    Evaluates phoneme clarity using ASR confidence.
    """

    def evaluate(self, vocals: np.ndarray,
                 expected_phonemes: List[str]) -> PhonemeMetrics:
        # Transcribe with ASR
        transcription = self.asr_model.transcribe(vocals)

        # Align and compare
        alignment = self._align_phonemes(
            transcription.phonemes,
            expected_phonemes
        )

        # Calculate metrics
        correct = sum(1 for a, e in alignment if a == e)
        substitutions = sum(1 for a, e in alignment if a != e and a and e)
        deletions = sum(1 for a, e in alignment if not a and e)
        insertions = sum(1 for a, e in alignment if a and not e)

        return PhonemeMetrics(
            phoneme_error_rate=(substitutions + deletions + insertions) / len(expected_phonemes),
            substitution_rate=substitutions / len(expected_phonemes),
            clarity_score=correct / len(expected_phonemes),
            confused_phonemes=self._find_confusions(alignment),
        )
```

### 6.5 Pass/Fail Thresholds

| Category | Metric | Alpha | Beta | Production |
|----------|--------|-------|------|------------|
| Pitch | Accuracy (cents) | ≤ 40 | ≤ 30 | ≤ 25 |
| Pitch | Intonation Score | ≥ 90% | ≥ 93% | ≥ 95% |
| Timing | Onset Error (ms) | ≤ 50 | ≤ 40 | ≤ 30 |
| Phoneme | Error Rate | ≤ 10% | ≤ 6% | ≤ 4% |
| Phoneme | Clarity Score | ≥ 85% | ≥ 92% | ≥ 96% |
| Timbre | Identity Drift | ≤ 0.15 | ≤ 0.10 | ≤ 0.05 |
| Overall | MOS (Human) | ≥ 3.5 | ≥ 4.0 | ≥ 4.3 |

### 6.6 Human Evaluation Rubric

```markdown
## Vocal Quality Evaluation Form

**Evaluator ID:** ____________  **Sample ID:** ____________

### 1. Naturalness (1-5)
How natural does the voice sound? (1=robotic, 5=human)
[ ] 1  [ ] 2  [ ] 3  [ ] 4  [ ] 5

### 2. Intelligibility (1-5)
How easy is it to understand the lyrics? (1=unintelligible, 5=crystal clear)
[ ] 1  [ ] 2  [ ] 3  [ ] 4  [ ] 5

### 3. Pitch Accuracy (1-5)
Is the singer on pitch? (1=very off, 5=perfect)
[ ] 1  [ ] 2  [ ] 3  [ ] 4  [ ] 5

### 4. Expression (1-5)
Does the performance convey appropriate emotion? (1=flat, 5=moving)
[ ] 1  [ ] 2  [ ] 3  [ ] 4  [ ] 5

### 5. Genre Fit (1-5)
Does the vocal style match the genre? (1=wrong style, 5=perfect fit)
[ ] 1  [ ] 2  [ ] 3  [ ] 4  [ ] 5

### 6. Overall Quality (1-5)
Overall impression of vocal quality (1=poor, 5=professional)
[ ] 1  [ ] 2  [ ] 3  [ ] 4  [ ] 5

### 7. Would you use this in a final mix?
[ ] Yes  [ ] With edits  [ ] No

### Comments:
_________________________________________________
```

---

## 7. Training & Data Strategy

### 7.1 Data Requirements (Legal-Safe)

| Data Type | Purpose | Quantity | Source Requirements |
|-----------|---------|----------|---------------------|
| Clean Vocals | Base voice modeling | 50-100 hours | Licensed studio sessions |
| Singing Exercises | Range/technique coverage | 20 hours | Original recordings |
| Multilingual Samples | Language phonetics | 30 hours per language | Native professional singers |
| Genre Exemplars | Style reference (not imitation) | 10 hours per genre | Licensed or public domain |
| Emotional Performances | Expression modeling | 15 hours | Directed studio sessions |

### 7.2 Originality Assurance

```python
class VoiceOriginalityGuard:
    """
    Ensures synthesized voice doesn't resemble real artists.
    """

    def __init__(self):
        # Load reference embeddings of protected artists
        self.protected_embeddings = self._load_protected_artists()
        self.similarity_threshold = 0.75  # Max allowed similarity

    def check_originality(self, synthesized_voice: np.ndarray) -> OriginalityReport:
        """
        Compare synthesized voice against protected references.
        """
        synth_embedding = self.encoder.encode(synthesized_voice)

        similarities = []
        for artist_id, artist_embedding in self.protected_embeddings.items():
            sim = cosine_similarity(synth_embedding, artist_embedding)
            similarities.append((artist_id, sim))

        max_similarity = max(similarities, key=lambda x: x[1])

        return OriginalityReport(
            is_original=max_similarity[1] < self.similarity_threshold,
            closest_match=max_similarity[0],
            similarity_score=max_similarity[1],
            all_comparisons=similarities
        )

    def validate_training_data(self, audio_files: List[Path]) -> DataValidationReport:
        """
        Ensure training data doesn't contain protected content.
        """
        issues = []
        for file in audio_files:
            # Check for known copyrighted works
            fingerprint = self._compute_fingerprint(file)
            matches = self.content_id_db.search(fingerprint)
            if matches:
                issues.append(CopyrightIssue(file=file, matches=matches))

        return DataValidationReport(
            files_checked=len(audio_files),
            issues=issues,
            approved=len(issues) == 0
        )
```

### 7.3 Consistency Maintenance

```python
class VoiceConsistencyMonitor:
    """
    Monitors for identity drift across model updates.
    """

    def __init__(self, reference_samples: List[np.ndarray]):
        self.reference_embeddings = [
            self.encoder.encode(s) for s in reference_samples
        ]
        self.reference_centroid = np.mean(self.reference_embeddings, axis=0)

    def check_consistency(self, new_model: SingingModel) -> ConsistencyReport:
        """
        Verify new model maintains voice identity.
        """
        # Generate same test phrases with new model
        test_outputs = [new_model.synthesize(p) for p in self.test_phrases]
        new_embeddings = [self.encoder.encode(o) for o in test_outputs]
        new_centroid = np.mean(new_embeddings, axis=0)

        # Calculate drift
        drift = np.linalg.norm(new_centroid - self.reference_centroid)

        return ConsistencyReport(
            identity_drift=drift,
            max_allowed_drift=0.05,
            is_consistent=drift < 0.05,
            per_phrase_drift=[
                np.linalg.norm(n - r)
                for n, r in zip(new_embeddings, self.reference_embeddings)
            ]
        )
```

### 7.4 Anti-Imprinting Measures

```python
class ImpressionPrevention:
    """
    Prevents model from memorizing specific artist characteristics.
    """

    SAFEGUARDS = {
        "data_mixing": {
            "description": "Never train on single-artist batches",
            "implementation": "Mix 5+ singers per batch minimum",
        },
        "style_normalization": {
            "description": "Normalize stylistic features across training",
            "implementation": "Remove artist-identifying ornaments during preprocessing",
        },
        "embedding_regularization": {
            "description": "Penalize similarity to protected artists",
            "implementation": "Add loss term for protected artist distance",
        },
        "periodic_audit": {
            "description": "Regular similarity checks against artist database",
            "implementation": "Weekly automated comparison, block if > 0.75 similarity",
        },
    }
```

---

## 8. Roadmap & Milestones

### 8.1 Version 0.1: Internal Alpha Singer

**Target:** Internal testing only

**Capabilities:**
- Single language (English)
- 3 genres (Pop, Rock, R&B)
- Lead vocal only (no harmonies)
- Basic vibrato and pitch control
- Fixed vocal identity

**Known Limitations:**
- Limited dynamic range
- No real-time processing
- Manual lyric alignment required
- No ornamentation system
- English pronunciation only

**Definition of Done:**
- [ ] Pitch accuracy ≤ 40 cents mean error
- [ ] Phoneme clarity ≥ 85%
- [ ] MOS ≥ 3.5 from internal team
- [ ] No similarity > 0.75 to protected artists
- [ ] Passes basic genre fit test (3/3 genres)

### 8.2 Version 0.2: Bilingual + Genre-Aware

**Target:** Expanded internal testing

**Capabilities:**
- Bilingual (English + Spanish)
- 8 genres with style adaptation
- Automatic lyric alignment
- Basic harmony generation (3rds, 5ths)
- Ornamentation system (runs, bends)
- Emotion curve support

**Known Limitations:**
- No whisper/breathy modes
- Limited harmony voicings
- Some artifacts in fast passages
- No real-time capability
- No code-switching mid-phrase

**Definition of Done:**
- [ ] Pitch accuracy ≤ 30 cents
- [ ] Phoneme clarity ≥ 92% (both languages)
- [ ] MOS ≥ 4.0 from expanded panel
- [ ] Passes bilingual intelligibility test
- [ ] Genre authenticity ≥ 4.0/5.0 (8/8 genres)
- [ ] Harmony accuracy ≥ 95% correct intervals

### 8.3 Version 1.0: Production-Grade Flagship Vocalist

**Target:** Commercial release

**Capabilities:**
- Full bilingual with code-switching
- 18 genres with authentic delivery
- Full vocal arrangement system
- Whisper, doubles, stacks, harmonies
- Ad-lib generation
- Real-time preview mode
- Emotion-responsive performance
- Custom style fine-tuning API

**Known Limitations:**
- No additional languages (Spanish/English only)
- No voice customization (fixed identity)
- Max 4-minute continuous generation
- Requires post-production for final master

**Definition of Done:**
- [ ] Pitch accuracy ≤ 25 cents
- [ ] Phoneme clarity ≥ 96% (both languages)
- [ ] MOS ≥ 4.3 from public panel (n=100)
- [ ] Genre authenticity ≥ 4.5/5.0 (18/18 genres)
- [ ] A/B test: ≥ 40% rate as "possibly human"
- [ ] No similarity > 0.70 to any protected artist
- [ ] Latency ≤ 2x real-time for preview
- [ ] Integration tests pass (100% AETHER pipeline)
- [ ] Legal review approved
- [ ] Professional producer sign-off

---

## 9. Risk Controls

### 9.1 "Generic AI Singer" Syndrome

**Risk:** Voice sounds competent but forgettable, lacking distinctive character.

**Mitigations:**
- Define and enforce identity invariants (Section 1.5)
- Regular "blind comparison" tests with human singers
- Reviewer checklist: "Would you remember this voice?"
- Avoid over-smoothing in post-processing
- Preserve intentional imperfections

**Detection:**
```python
def detect_genericism(samples: List[np.ndarray]) -> float:
    """
    Measure distinctiveness vs. average AI voice.
    Higher = more distinctive.
    """
    sample_embeddings = [encoder.encode(s) for s in samples]
    generic_centroid = load_generic_ai_voice_centroid()

    distances = [
        np.linalg.norm(e - generic_centroid)
        for e in sample_embeddings
    ]
    return np.mean(distances)  # Target: > 0.3
```

### 9.2 Over-Polished / Emotionless Vocals

**Risk:** Perfect pitch and timing but no soul.

**Mitigations:**
- Intentional micro-timing variations (±10ms jitter)
- Dynamic expression curves tied to lyrics
- Breath and effort sounds on high notes
- Imperfect transitions (subtle scoops/falls)
- Emotion evaluator in QC pipeline

**Detection:**
- Human evaluation: "Does this feel expressive?" (≥ 4.0/5.0)
- Variance in performance parameters (should not be 0)
- Correlation between lyrics sentiment and vocal intensity

### 9.3 Genre Collapse at Vocal Level

**Risk:** All genres sound the same vocally despite instrumental differences.

**Mitigations:**
- Explicit genre performance profiles (Section 4.2)
- Genre-specific ornamentation rules
- Automated genre classification test on vocals alone
- A/B testing: "Which genre is this?" accuracy ≥ 85%

**Detection:**
```python
def test_genre_distinction(model: SingingModel,
                           test_lyrics: str,
                           genres: List[str]) -> float:
    """
    Test if same lyrics sound different across genres.
    """
    outputs = {g: model.synthesize(test_lyrics, genre=g) for g in genres}
    embeddings = {g: encoder.encode(o) for g, o in outputs.items()}

    # Calculate inter-genre distances
    distances = []
    for g1 in genres:
        for g2 in genres:
            if g1 < g2:
                dist = np.linalg.norm(embeddings[g1] - embeddings[g2])
                distances.append(dist)

    return np.mean(distances)  # Target: > 0.2
```

### 9.4 Multilingual Pronunciation Degradation

**Risk:** Spanish suffers when English improves (or vice versa).

**Mitigations:**
- Separate phoneme models per language
- Balanced training batches (50/50 language split)
- Per-language evaluation in every release
- Native speaker review panel for each language

**Detection:**
- Track per-language metrics independently
- Alert if either language drops > 5% on any metric
- Quarterly native speaker blind evaluation

### 9.5 Vocal Identity Drift Over Time

**Risk:** Voice changes subtly with model updates until no longer recognizable.

**Mitigations:**
- Reference recordings frozen at v1.0
- Consistency check before every release (Section 7.3)
- Identity drift budget: max 0.05 per version
- Cumulative drift tracking across versions

**Detection:**
```python
class IdentityDriftTracker:
    def __init__(self, v1_reference: np.ndarray):
        self.v1_embedding = encoder.encode(v1_reference)
        self.drift_history = []

    def check_version(self, new_model: SingingModel) -> DriftReport:
        new_output = new_model.synthesize(self.test_phrase)
        new_embedding = encoder.encode(new_output)

        cumulative_drift = np.linalg.norm(new_embedding - self.v1_embedding)

        if cumulative_drift > 0.15:  # Total drift budget
            return DriftReport(
                status="BLOCKED",
                cumulative_drift=cumulative_drift,
                message="Cumulative identity drift exceeds budget"
            )

        self.drift_history.append(cumulative_drift)
        return DriftReport(status="PASSED", cumulative_drift=cumulative_drift)
```

---

## Appendix A: File Structure

```
src/aether/voice/
├── __init__.py
├── identity/
│   ├── blueprint.py          # Vocal identity definition
│   ├── invariants.py         # Identity constraints
│   └── drift_monitor.py      # Consistency tracking
├── phonetics/
│   ├── english.py            # English phoneme system
│   ├── spanish.py            # Spanish phoneme system
│   ├── prosody.py            # Language prosody rules
│   └── bilingual.py          # Code-switching handler
├── engine/
│   ├── aligner.py            # Lyric-melody alignment
│   ├── pitch.py              # Pitch contour generation
│   ├── vibrato.py            # Vibrato modeling
│   ├── transitions.py        # Note transition types
│   ├── breath.py             # Breath modeling
│   └── synthesizer.py        # Core neural vocoder
├── performance/
│   ├── profiles.py           # Genre performance profiles
│   ├── ornamentation.py      # Ornament generation
│   └── expression.py         # Emotion/dynamics curves
├── arrangement/
│   ├── layers.py             # Vocal layer types
│   ├── harmony.py            # Harmony generation
│   ├── stacking.py           # Vocal stacking
│   └── safeguards.py         # Phase/masking prevention
├── quality/
│   ├── metrics.py            # Automated metrics
│   ├── evaluator.py          # Quality analysis
│   └── thresholds.py         # Pass/fail criteria
└── integration/
    ├── pipeline.py           # AETHER integration
    └── api.py                # External API
```

---

**Document Version:** 1.0
**Classification:** Internal Engineering Specification
**Review Cycle:** Quarterly or upon major release
