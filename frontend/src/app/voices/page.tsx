'use client'

import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Loader2,
  Mic,
  User,
  Sliders,
  Plus,
  Trash2,
  Volume2,
  Save,
  ChevronDown,
  ChevronUp,
  Play,
  Pause,
  Square,
  RotateCcw,
  Sparkles,
  Waves,
  RefreshCw,
  Check,
  Music,
  Zap,
} from 'lucide-react'
import {
  aetherApi,
  Voice,
  VoiceTimbreParams,
  VoiceEmotionParams,
  VoiceVibratoParams,
  CustomVoiceRequest,
} from '@/lib/api'

// API base URL for audio
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// =============================================================================
// Types
// =============================================================================

interface SliderConfig {
  key: string
  label: string
  min: number
  max: number
  step: number
  unit?: string
  description: string
  icon?: React.ElementType
}

interface VoicePreset {
  name: string
  description: string
  icon: React.ElementType
  timbre: Partial<VoiceTimbreParams>
  emotion: Partial<VoiceEmotionParams>
}

// =============================================================================
// Professional Presets Library
// =============================================================================

const VOICE_PRESETS: VoicePreset[] = [
  {
    name: 'Pop Star',
    description: 'Bright, breathy, modern pop sound',
    icon: Sparkles,
    timbre: { brightness: 0.75, breathiness: 0.35, grit: 0.08, chest_resonance: 0.55 },
    emotion: { warmth: 0.65, intimacy: 0.70, engagement: 0.85 },
  },
  {
    name: 'R&B Soul',
    description: 'Warm, smooth, soulful delivery',
    icon: Music,
    timbre: { brightness: 0.55, breathiness: 0.28, grit: 0.15, chest_resonance: 0.72 },
    emotion: { warmth: 0.82, intimacy: 0.75, sincerity: 0.80 },
  },
  {
    name: 'Rock Power',
    description: 'Gritty, powerful, commanding',
    icon: Zap,
    timbre: { brightness: 0.60, breathiness: 0.15, grit: 0.35, chest_resonance: 0.85 },
    emotion: { warmth: 0.50, power_reserve: 0.90, engagement: 0.78 },
  },
  {
    name: 'Jazz Smooth',
    description: 'Dark, controlled, sophisticated',
    icon: Waves,
    timbre: { brightness: 0.42, breathiness: 0.22, grit: 0.12, head_voice_blend: 0.45 },
    emotion: { warmth: 0.70, control: 0.85, intimacy: 0.68 },
  },
  {
    name: 'Intimate Acoustic',
    description: 'Soft, close, vulnerable',
    icon: Mic,
    timbre: { brightness: 0.50, breathiness: 0.40, grit: 0.05, chest_resonance: 0.48 },
    emotion: { warmth: 0.75, intimacy: 0.90, sincerity: 0.88 },
  },
]

// =============================================================================
// Slider Configurations
// =============================================================================

const TIMBRE_SLIDERS: SliderConfig[] = [
  { key: 'brightness', label: 'Brightness', min: 0, max: 1, step: 0.01, description: 'Tone color from dark/warm to bright/present' },
  { key: 'breathiness', label: 'Breathiness', min: 0, max: 1, step: 0.01, description: 'Amount of air in the tone for intimacy' },
  { key: 'grit', label: 'Grit', min: 0, max: 1, step: 0.01, description: 'Raspiness and character in the voice' },
  { key: 'nasality', label: 'Nasality', min: 0, max: 1, step: 0.01, description: 'Nasal resonance coloring' },
  { key: 'chest_resonance', label: 'Chest Resonance', min: 0, max: 1, step: 0.01, description: 'Lower body and chest voice strength' },
  { key: 'head_voice_blend', label: 'Head Voice', min: 0, max: 1, step: 0.01, description: 'Upper register and head voice mix' },
]

const EMOTION_SLIDERS: SliderConfig[] = [
  { key: 'warmth', label: 'Warmth', min: 0, max: 1, step: 0.01, description: 'Overall vocal warmth and friendliness' },
  { key: 'control', label: 'Control', min: 0, max: 1, step: 0.01, description: 'Technical precision and steadiness' },
  { key: 'intimacy', label: 'Intimacy', min: 0, max: 1, step: 0.01, description: 'Closeness and personal connection' },
  { key: 'power_reserve', label: 'Power Reserve', min: 0, max: 1, step: 0.01, description: 'Dynamic headroom for climactic moments' },
  { key: 'sincerity', label: 'Sincerity', min: 0, max: 1, step: 0.01, description: 'Authenticity and believability' },
  { key: 'engagement', label: 'Engagement', min: 0, max: 1, step: 0.01, description: 'Active listener connection' },
]

const VIBRATO_SLIDERS: SliderConfig[] = [
  { key: 'rate_min', label: 'Rate Min', min: 3, max: 8, step: 0.1, unit: 'Hz', description: 'Minimum vibrato oscillation speed' },
  { key: 'rate_max', label: 'Rate Max', min: 3, max: 8, step: 0.1, unit: 'Hz', description: 'Maximum vibrato oscillation speed' },
  { key: 'onset_delay_min', label: 'Onset Min', min: 50, max: 500, step: 10, unit: 'ms', description: 'Minimum delay before vibrato starts' },
  { key: 'onset_delay_max', label: 'Onset Max', min: 50, max: 500, step: 10, unit: 'ms', description: 'Maximum delay before vibrato starts' },
]

// =============================================================================
// MIDI Note Helper
// =============================================================================

function midiToNote(midi: number): string {
  const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  const octave = Math.floor(midi / 12) - 1
  const note = notes[midi % 12]
  return `${note}${octave}`
}

// =============================================================================
// Waveform Visualization Component
// =============================================================================

function WaveformVisualizer({ isPlaying, progress }: { isPlaying: boolean; progress: number }) {
  const bars = 40

  return (
    <div className="flex items-center justify-center gap-0.5 h-12 px-2">
      {Array.from({ length: bars }).map((_, i) => {
        const isActive = (i / bars) * 100 <= progress
        const height = isPlaying
          ? Math.sin((i + Date.now() / 100) * 0.3) * 50 + 50
          : 30
        return (
          <div
            key={i}
            className={`w-1 rounded-full transition-all duration-75 ${
              isActive ? 'bg-purple-500' : 'bg-zinc-700'
            }`}
            style={{ height: `${isPlaying ? height : 30}%` }}
          />
        )
      })}
    </div>
  )
}

// =============================================================================
// Professional Parameter Slider
// =============================================================================

function ParameterSlider({
  config,
  value,
  originalValue,
  onChange,
  disabled = false,
}: {
  config: SliderConfig
  value: number
  originalValue?: number
  onChange: (value: number) => void
  disabled?: boolean
}) {
  const hasChanged = originalValue !== undefined && Math.abs(value - originalValue) > 0.001
  const displayValue = config.unit
    ? `${value.toFixed(config.step < 1 ? 1 : 0)}${config.unit}`
    : `${(value * 100).toFixed(0)}%`

  // Calculate position percentage for the filled track
  const percentage = ((value - config.min) / (config.max - config.min)) * 100

  return (
    <div className="space-y-2 group">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <label className={`text-sm font-medium transition-colors ${hasChanged ? 'text-purple-400' : 'text-zinc-300'}`}>
            {config.label}
          </label>
          {hasChanged && (
            <span className="text-xs text-purple-400 bg-purple-400/10 px-1.5 py-0.5 rounded">
              modified
            </span>
          )}
        </div>
        <span className={`text-sm font-mono tabular-nums ${hasChanged ? 'text-purple-400' : 'text-zinc-500'}`}>
          {displayValue}
        </span>
      </div>

      <div className="relative">
        <div className="absolute inset-0 h-2 bg-zinc-700 rounded-full" />
        <div
          className="absolute h-2 bg-gradient-to-r from-purple-600 to-purple-400 rounded-full transition-all"
          style={{ width: `${percentage}%` }}
        />
        <input
          type="range"
          min={config.min}
          max={config.max}
          step={config.step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          disabled={disabled}
          className="relative w-full h-2 bg-transparent appearance-none cursor-pointer z-10
                     disabled:opacity-50 disabled:cursor-not-allowed
                     [&::-webkit-slider-thumb]:appearance-none
                     [&::-webkit-slider-thumb]:w-4
                     [&::-webkit-slider-thumb]:h-4
                     [&::-webkit-slider-thumb]:rounded-full
                     [&::-webkit-slider-thumb]:bg-white
                     [&::-webkit-slider-thumb]:shadow-lg
                     [&::-webkit-slider-thumb]:cursor-pointer
                     [&::-webkit-slider-thumb]:transition-transform
                     [&::-webkit-slider-thumb]:hover:scale-110
                     [&::-webkit-slider-thumb]:active:scale-95"
        />
      </div>

      <p className="text-xs text-zinc-500 opacity-0 group-hover:opacity-100 transition-opacity">
        {config.description}
      </p>
    </div>
  )
}

// =============================================================================
// Audio Player with Waveform
// =============================================================================

function ProfessionalAudioPlayer({
  voiceName,
  label,
  audioUrl,
  onPlay,
  isActive,
}: {
  voiceName: string
  label?: string
  audioUrl?: string
  onPlay?: () => void
  isActive?: boolean
}) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [duration, setDuration] = useState(0)
  const animationRef = useRef<number>()

  const url = audioUrl || `${API_BASE_URL}/v1/voices/${encodeURIComponent(voiceName)}/audio?t=${Date.now()}`

  useEffect(() => {
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [])

  useEffect(() => {
    // Stop if not active
    if (!isActive && isPlaying && audioRef.current) {
      audioRef.current.pause()
      setIsPlaying(false)
    }
  }, [isActive, isPlaying])

  const animate = useCallback(() => {
    if (audioRef.current) {
      setProgress((audioRef.current.currentTime / audioRef.current.duration) * 100 || 0)
      animationRef.current = requestAnimationFrame(animate)
    }
  }, [])

  const handlePlay = async () => {
    if (!audioRef.current) return

    if (isPlaying) {
      audioRef.current.pause()
      setIsPlaying(false)
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      return
    }

    setIsLoading(true)
    try {
      audioRef.current.src = url
      await audioRef.current.play()
      setIsPlaying(true)
      setDuration(audioRef.current.duration)
      animate()
      onPlay?.()
    } catch (err) {
      console.error('Playback failed:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleEnded = () => {
    setIsPlaying(false)
    setProgress(0)
    if (animationRef.current) cancelAnimationFrame(animationRef.current)
  }

  return (
    <div className={`rounded-xl border transition-all ${
      isActive ? 'border-purple-500 bg-purple-500/5' : 'border-zinc-700 bg-zinc-800/50'
    }`}>
      <audio ref={audioRef} onEnded={handleEnded} />

      {label && (
        <div className="px-4 py-2 border-b border-zinc-700/50">
          <span className="text-xs font-medium text-zinc-400 uppercase tracking-wider">{label}</span>
        </div>
      )}

      <div className="p-4">
        <WaveformVisualizer isPlaying={isPlaying} progress={progress} />

        <div className="flex items-center justify-between mt-3">
          <button
            onClick={handlePlay}
            disabled={isLoading}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
              isPlaying
                ? 'bg-purple-500 text-white'
                : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
            } disabled:opacity-50`}
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : isPlaying ? (
              <Pause className="w-4 h-4" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            {isPlaying ? 'Pause' : 'Play'}
          </button>

          <span className="text-xs text-zinc-500 font-mono">
            {Math.floor(progress / 100 * duration || 0)}s / {Math.floor(duration || 0)}s
          </span>
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// Voice Card Component
// =============================================================================

function VoiceCard({
  voice,
  isSelected,
  onSelect,
  onPlayPreview,
  isCurrentlyPlaying,
}: {
  voice: Voice
  isSelected: boolean
  onSelect: () => void
  onPlayPreview: (name: string) => void
  isCurrentlyPlaying: boolean
}) {
  const isCustom = voice.name.startsWith('custom-') || voice.character.includes('Custom')
  const audioRef = useRef<HTMLAudioElement>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handlePlayClick = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!audioRef.current) return

    if (isCurrentlyPlaying) {
      audioRef.current.pause()
      onPlayPreview('')
      return
    }

    setIsLoading(true)
    try {
      audioRef.current.src = `${API_BASE_URL}/v1/voices/${encodeURIComponent(voice.name)}/audio?t=${Date.now()}`
      await audioRef.current.play()
      onPlayPreview(voice.name)
    } catch (err) {
      console.error('Playback failed:', err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div
      onClick={onSelect}
      className={`w-full p-4 rounded-xl border transition-all text-left cursor-pointer
        ${isSelected
          ? 'border-purple-500 bg-purple-500/10 shadow-lg shadow-purple-500/10'
          : 'border-zinc-700 bg-zinc-800/50 hover:border-zinc-600 hover:bg-zinc-800'
        }`}
    >
      <audio ref={audioRef} onEnded={() => onPlayPreview('')} />

      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center transition-colors
            ${isSelected ? 'bg-purple-500' : 'bg-zinc-700'}`}>
            {isCustom ? <Sliders className="w-5 h-5" /> : <User className="w-5 h-5" />}
          </div>
          <div>
            <h3 className="font-semibold text-white">{voice.name}</h3>
            <p className="text-sm text-zinc-400">{voice.classification}</p>
          </div>
        </div>

        <button
          onClick={handlePlayClick}
          className={`p-2.5 rounded-full transition-all ${
            isCurrentlyPlaying
              ? 'bg-purple-500 text-white scale-110'
              : 'bg-zinc-600 text-zinc-300 hover:bg-zinc-500 hover:scale-105'
          }`}
          title={isCurrentlyPlaying ? 'Stop' : 'Play preview'}
        >
          {isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : isCurrentlyPlaying ? (
            <Square className="w-4 h-4" />
          ) : (
            <Play className="w-4 h-4" />
          )}
        </button>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <span className="px-2 py-0.5 bg-zinc-700 rounded text-xs text-zinc-300 font-mono">
          {midiToNote(voice.range_low)} - {midiToNote(voice.range_high)}
        </span>
        {voice.character.split(', ').slice(0, 3).map((trait) => (
          <span key={trait} className="px-2 py-0.5 bg-zinc-700/50 rounded text-xs text-zinc-400">
            {trait}
          </span>
        ))}
      </div>
    </div>
  )
}

// =============================================================================
// Parameter Section with Collapse
// =============================================================================

function ParameterSection({
  title,
  icon: Icon,
  children,
  defaultOpen = true,
  badge,
}: {
  title: string
  icon: React.ElementType
  children: React.ReactNode
  defaultOpen?: boolean
  badge?: string
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="border border-zinc-700 rounded-xl overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between bg-zinc-800/50 hover:bg-zinc-800 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-purple-400" />
          <span className="font-medium">{title}</span>
          {badge && (
            <span className="text-xs bg-purple-500/20 text-purple-400 px-2 py-0.5 rounded-full">
              {badge}
            </span>
          )}
        </div>
        {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      {isOpen && (
        <div className="p-4 space-y-5 bg-zinc-900/50">
          {children}
        </div>
      )}
    </div>
  )
}

// =============================================================================
// Preset Selector
// =============================================================================

function PresetSelector({
  onApply,
}: {
  onApply: (preset: VoicePreset) => void
}) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
      {VOICE_PRESETS.map((preset) => (
        <button
          key={preset.name}
          onClick={() => onApply(preset)}
          className="p-3 rounded-lg border border-zinc-700 bg-zinc-800/50 hover:border-purple-500/50
                     hover:bg-purple-500/5 transition-all text-left group"
        >
          <preset.icon className="w-5 h-5 text-zinc-500 group-hover:text-purple-400 mb-2 transition-colors" />
          <h4 className="font-medium text-sm text-white">{preset.name}</h4>
          <p className="text-xs text-zinc-500 mt-0.5 line-clamp-2">{preset.description}</p>
        </button>
      ))}
    </div>
  )
}

// =============================================================================
// Main Page Component
// =============================================================================

export default function VoicesPage() {
  const queryClient = useQueryClient()
  const [selectedVoice, setSelectedVoice] = useState<Voice | null>(null)
  const [isCreatingCustom, setIsCreatingCustom] = useState(false)
  const [customName, setCustomName] = useState('')
  const [playingVoice, setPlayingVoice] = useState<string>('')
  const [isPreviewingChanges, setIsPreviewingChanges] = useState(false)
  const [customPreviewUrl, setCustomPreviewUrl] = useState<string | null>(null)
  const [activePlayer, setActivePlayer] = useState<'original' | 'modified' | null>(null)

  // Edited parameters
  const [editedTimbre, setEditedTimbre] = useState<VoiceTimbreParams | null>(null)
  const [editedEmotion, setEditedEmotion] = useState<VoiceEmotionParams | null>(null)
  const [editedVibrato, setEditedVibrato] = useState<VoiceVibratoParams | null>(null)

  // Fetch voices
  const { data: voicesData, isLoading, error } = useQuery({
    queryKey: ['voices'],
    queryFn: () => aetherApi.listVoices(),
  })

  // Create custom voice mutation
  const createCustomMutation = useMutation({
    mutationFn: (params: CustomVoiceRequest) => aetherApi.createCustomVoice(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['voices'] })
      setIsCreatingCustom(false)
      setCustomName('')
      resetEdits()
    },
  })

  // Delete custom voice mutation
  const deleteCustomMutation = useMutation({
    mutationFn: (voiceId: string) => aetherApi.deleteCustomVoice(voiceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['voices'] })
      setSelectedVoice(null)
    },
  })

  // Select voice and initialize edit state
  const handleSelectVoice = useCallback((voice: Voice) => {
    setSelectedVoice(voice)
    setEditedTimbre({ ...voice.timbre })
    setEditedEmotion({ ...voice.emotion })
    setEditedVibrato({ ...voice.vibrato })
    setCustomPreviewUrl(null)
    setActivePlayer(null)
  }, [])

  // Reset edits
  const resetEdits = useCallback(() => {
    if (selectedVoice) {
      setEditedTimbre({ ...selectedVoice.timbre })
      setEditedEmotion({ ...selectedVoice.emotion })
      setEditedVibrato({ ...selectedVoice.vibrato })
      setCustomPreviewUrl(null)
    }
  }, [selectedVoice])

  // Apply preset
  const applyPreset = useCallback((preset: VoicePreset) => {
    if (!editedTimbre || !editedEmotion) return

    setEditedTimbre({ ...editedTimbre, ...preset.timbre })
    setEditedEmotion({ ...editedEmotion, ...preset.emotion })
    setCustomPreviewUrl(null)
  }, [editedTimbre, editedEmotion])

  // Preview changes - generate custom audio
  const handlePreviewChanges = useCallback(async () => {
    if (!selectedVoice || !editedTimbre || !editedEmotion || !editedVibrato) return

    setIsPreviewingChanges(true)
    try {
      const response = await fetch(`${API_BASE_URL}/v1/voices/preview-custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_voice: selectedVoice.name,
          timbre: editedTimbre,
          emotion: editedEmotion,
          vibrato: editedVibrato,
          backend: 'auto',
        }),
      })

      if (!response.ok) throw new Error('Preview failed')

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      setCustomPreviewUrl(url)
      setActivePlayer('modified')
    } catch (err) {
      console.error('Preview generation failed:', err)
    } finally {
      setIsPreviewingChanges(false)
    }
  }, [selectedVoice, editedTimbre, editedEmotion, editedVibrato])

  // Save custom voice
  const handleSaveCustom = useCallback(() => {
    if (!selectedVoice || !customName.trim()) return

    createCustomMutation.mutate({
      name: customName.trim(),
      base_voice: selectedVoice.name,
      timbre: editedTimbre || undefined,
      emotion: editedEmotion || undefined,
      vibrato: editedVibrato || undefined,
    })
  }, [selectedVoice, customName, editedTimbre, editedEmotion, editedVibrato, createCustomMutation])

  // Check for changes
  const hasChanges = useMemo(() => {
    if (!selectedVoice || !editedTimbre) return false
    return (
      JSON.stringify(editedTimbre) !== JSON.stringify(selectedVoice.timbre) ||
      JSON.stringify(editedEmotion) !== JSON.stringify(selectedVoice.emotion) ||
      JSON.stringify(editedVibrato) !== JSON.stringify(selectedVoice.vibrato)
    )
  }, [selectedVoice, editedTimbre, editedEmotion, editedVibrato])

  // Count modified parameters
  const modifiedCount = useMemo(() => {
    if (!selectedVoice || !editedTimbre || !editedEmotion || !editedVibrato) return 0
    let count = 0
    Object.keys(editedTimbre).forEach(key => {
      if (Math.abs((editedTimbre as any)[key] - (selectedVoice.timbre as any)[key]) > 0.001) count++
    })
    Object.keys(editedEmotion).forEach(key => {
      if (Math.abs((editedEmotion as any)[key] - (selectedVoice.emotion as any)[key]) > 0.001) count++
    })
    return count
  }, [selectedVoice, editedTimbre, editedEmotion, editedVibrato])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-zinc-900 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-purple-500 mx-auto mb-4" />
          <p className="text-zinc-400">Loading voices...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-zinc-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 mb-2">Failed to load voices</p>
          <p className="text-zinc-500 text-sm">{(error as Error).message}</p>
        </div>
      </div>
    )
  }

  const voices = voicesData?.voices || []

  return (
    <div className="min-h-screen bg-zinc-900 text-white">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4 sticky top-0 bg-zinc-900/95 backdrop-blur z-10">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <Mic className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-semibold">Voice Studio</h1>
              <p className="text-xs text-zinc-500">Professional voice fine-tuning</p>
            </div>
          </div>
          <span className="text-sm text-zinc-500 bg-zinc-800 px-3 py-1 rounded-full">
            {voices.length} voices
          </span>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Voice List */}
          <div className="lg:col-span-1 space-y-4">
            <h2 className="text-lg font-semibold text-zinc-300">Select Voice</h2>
            <div className="space-y-3">
              {voices.map((voice) => (
                <VoiceCard
                  key={voice.name}
                  voice={voice}
                  isSelected={selectedVoice?.name === voice.name}
                  onSelect={() => handleSelectVoice(voice)}
                  onPlayPreview={setPlayingVoice}
                  isCurrentlyPlaying={playingVoice === voice.name}
                />
              ))}
            </div>
          </div>

          {/* Voice Details & Fine-tuning */}
          <div className="lg:col-span-2 space-y-6">
            {selectedVoice ? (
              <>
                {/* Voice Header */}
                <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-2xl p-6 border border-purple-500/20">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h2 className="text-2xl font-bold">{selectedVoice.name}</h2>
                      <p className="text-zinc-400">{selectedVoice.classification}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      {hasChanges && (
                        <span className="text-xs bg-purple-500/20 text-purple-400 px-2 py-1 rounded-full">
                          {modifiedCount} changes
                        </span>
                      )}
                      {selectedVoice.name.startsWith('custom-') && (
                        <button
                          onClick={() => deleteCustomMutation.mutate(selectedVoice.name)}
                          className="p-2 text-red-400 hover:bg-red-400/10 rounded-lg transition-colors"
                          title="Delete custom voice"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Range Info */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-zinc-800/50 rounded-lg p-3">
                      <span className="text-zinc-500 text-xs">Comfortable Range</span>
                      <p className="font-mono text-lg">
                        {midiToNote(selectedVoice.range_low)} - {midiToNote(selectedVoice.range_high)}
                      </p>
                    </div>
                    <div className="bg-zinc-800/50 rounded-lg p-3">
                      <span className="text-zinc-500 text-xs">Tessitura</span>
                      <p className="font-mono text-lg">
                        {midiToNote(selectedVoice.tessitura_low)} - {midiToNote(selectedVoice.tessitura_high)}
                      </p>
                    </div>
                  </div>
                </div>

                {/* A/B Comparison Players */}
                {hasChanges && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <ProfessionalAudioPlayer
                      voiceName={selectedVoice.name}
                      label="Original"
                      isActive={activePlayer === 'original'}
                      onPlay={() => setActivePlayer('original')}
                    />
                    <div className="relative">
                      {customPreviewUrl ? (
                        <ProfessionalAudioPlayer
                          voiceName={selectedVoice.name}
                          label="Modified"
                          audioUrl={customPreviewUrl}
                          isActive={activePlayer === 'modified'}
                          onPlay={() => setActivePlayer('modified')}
                        />
                      ) : (
                        <div className="rounded-xl border border-dashed border-zinc-700 bg-zinc-800/30 p-6 flex flex-col items-center justify-center h-full min-h-[140px]">
                          <RefreshCw className="w-6 h-6 text-zinc-600 mb-2" />
                          <p className="text-sm text-zinc-500 text-center">
                            Click "Preview Changes" to hear your modifications
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Preview Changes Button */}
                {hasChanges && (
                  <button
                    onClick={handlePreviewChanges}
                    disabled={isPreviewingChanges}
                    className="w-full py-3 px-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500
                               rounded-xl font-semibold flex items-center justify-center gap-2 transition-all
                               disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-purple-500/20"
                  >
                    {isPreviewingChanges ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Generating Preview...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-5 h-5" />
                        Preview Changes
                      </>
                    )}
                  </button>
                )}

                {/* Preset Library */}
                <div className="space-y-3">
                  <h3 className="text-sm font-medium text-zinc-400 flex items-center gap-2">
                    <Sparkles className="w-4 h-4" />
                    Quick Presets
                  </h3>
                  <PresetSelector onApply={applyPreset} />
                </div>

                {/* Fine-tuning Controls */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                      <Sliders className="w-5 h-5 text-purple-400" />
                      Fine-tune Parameters
                    </h3>
                    {hasChanges && (
                      <button
                        onClick={resetEdits}
                        className="text-sm text-zinc-400 hover:text-white flex items-center gap-1 transition-colors"
                      >
                        <RotateCcw className="w-4 h-4" />
                        Reset
                      </button>
                    )}
                  </div>

                  {/* Timbre */}
                  <ParameterSection
                    title="Timbre"
                    icon={Volume2}
                    badge={editedTimbre && Object.keys(editedTimbre).filter(k =>
                      Math.abs((editedTimbre as any)[k] - (selectedVoice.timbre as any)[k]) > 0.001
                    ).length > 0 ? 'modified' : undefined}
                  >
                    {editedTimbre && TIMBRE_SLIDERS.map((config) => (
                      <ParameterSlider
                        key={config.key}
                        config={config}
                        value={editedTimbre[config.key as keyof VoiceTimbreParams]}
                        originalValue={selectedVoice.timbre[config.key as keyof VoiceTimbreParams]}
                        onChange={(v) => {
                          setEditedTimbre({ ...editedTimbre, [config.key]: v })
                          setCustomPreviewUrl(null)
                        }}
                      />
                    ))}
                  </ParameterSection>

                  {/* Emotion */}
                  <ParameterSection
                    title="Emotional Baseline"
                    icon={User}
                    defaultOpen={false}
                    badge={editedEmotion && Object.keys(editedEmotion).filter(k =>
                      Math.abs((editedEmotion as any)[k] - (selectedVoice.emotion as any)[k]) > 0.001
                    ).length > 0 ? 'modified' : undefined}
                  >
                    {editedEmotion && EMOTION_SLIDERS.map((config) => (
                      <ParameterSlider
                        key={config.key}
                        config={config}
                        value={editedEmotion[config.key as keyof VoiceEmotionParams]}
                        originalValue={selectedVoice.emotion[config.key as keyof VoiceEmotionParams]}
                        onChange={(v) => {
                          setEditedEmotion({ ...editedEmotion, [config.key]: v })
                          setCustomPreviewUrl(null)
                        }}
                      />
                    ))}
                  </ParameterSection>

                  {/* Vibrato */}
                  <ParameterSection title="Vibrato" icon={Waves} defaultOpen={false}>
                    {editedVibrato && VIBRATO_SLIDERS.map((config) => (
                      <ParameterSlider
                        key={config.key}
                        config={config}
                        value={editedVibrato[config.key as keyof VoiceVibratoParams]}
                        originalValue={selectedVoice.vibrato[config.key as keyof VoiceVibratoParams]}
                        onChange={(v) => {
                          setEditedVibrato({ ...editedVibrato, [config.key]: v })
                          setCustomPreviewUrl(null)
                        }}
                      />
                    ))}
                  </ParameterSection>
                </div>

                {/* Save as Custom */}
                {hasChanges && (
                  <div className="bg-zinc-800/50 border border-zinc-700 rounded-xl p-5">
                    {isCreatingCustom ? (
                      <div className="space-y-3">
                        <label className="text-sm font-medium text-zinc-300">Custom Voice Name</label>
                        <input
                          type="text"
                          value={customName}
                          onChange={(e) => setCustomName(e.target.value)}
                          placeholder="My Custom Voice..."
                          className="w-full px-4 py-3 bg-zinc-900 border border-zinc-600 rounded-lg
                                     focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500
                                     placeholder:text-zinc-600"
                        />
                        <div className="flex gap-2">
                          <button
                            onClick={handleSaveCustom}
                            disabled={!customName.trim() || createCustomMutation.isPending}
                            className="flex-1 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-medium
                                       disabled:opacity-50 disabled:cursor-not-allowed
                                       flex items-center justify-center gap-2 transition-colors"
                          >
                            {createCustomMutation.isPending ? (
                              <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                              <Check className="w-4 h-4" />
                            )}
                            Save Voice
                          </button>
                          <button
                            onClick={() => setIsCreatingCustom(false)}
                            className="px-6 py-3 bg-zinc-700 hover:bg-zinc-600 rounded-lg transition-colors"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <button
                        onClick={() => setIsCreatingCustom(true)}
                        className="w-full py-3 flex items-center justify-center gap-2
                                   text-purple-400 hover:text-purple-300 font-medium transition-colors"
                      >
                        <Save className="w-5 h-5" />
                        Save as Custom Voice
                      </button>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="bg-zinc-800/30 rounded-2xl p-16 border border-dashed border-zinc-700
                              flex flex-col items-center justify-center text-center">
                <div className="w-16 h-16 rounded-2xl bg-zinc-800 flex items-center justify-center mb-6">
                  <Mic className="w-8 h-8 text-zinc-600" />
                </div>
                <h3 className="text-xl font-semibold text-zinc-300 mb-2">Select a Voice</h3>
                <p className="text-zinc-500 max-w-md">
                  Choose a voice from the list to view its details, preview audio, and fine-tune parameters
                  to create your perfect custom voice.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
