'use client'

import { useState, useCallback } from 'react'
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
} from 'lucide-react'
import {
  aetherApi,
  Voice,
  VoiceTimbreParams,
  VoiceEmotionParams,
  VoiceVibratoParams,
  CustomVoiceRequest,
} from '@/lib/api'

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
}

// =============================================================================
// Slider Configurations
// =============================================================================

const TIMBRE_SLIDERS: SliderConfig[] = [
  { key: 'brightness', label: 'Brightness', min: 0, max: 1, step: 0.01, description: 'Tone brightness (dark to bright)' },
  { key: 'breathiness', label: 'Breathiness', min: 0, max: 1, step: 0.01, description: 'Air in tone' },
  { key: 'grit', label: 'Grit', min: 0, max: 1, step: 0.01, description: 'Raspiness/character' },
  { key: 'nasality', label: 'Nasality', min: 0, max: 1, step: 0.01, description: 'Nasal resonance' },
  { key: 'chest_resonance', label: 'Chest Resonance', min: 0, max: 1, step: 0.01, description: 'Chest voice strength' },
  { key: 'head_voice_blend', label: 'Head Voice', min: 0, max: 1, step: 0.01, description: 'Head voice mix' },
]

const EMOTION_SLIDERS: SliderConfig[] = [
  { key: 'warmth', label: 'Warmth', min: 0, max: 1, step: 0.01, description: 'Vocal warmth' },
  { key: 'control', label: 'Control', min: 0, max: 1, step: 0.01, description: 'Technical precision' },
  { key: 'intimacy', label: 'Intimacy', min: 0, max: 1, step: 0.01, description: 'Closeness feel' },
  { key: 'power_reserve', label: 'Power Reserve', min: 0, max: 1, step: 0.01, description: 'Dynamic headroom' },
  { key: 'sincerity', label: 'Sincerity', min: 0, max: 1, step: 0.01, description: 'Authenticity' },
  { key: 'engagement', label: 'Engagement', min: 0, max: 1, step: 0.01, description: 'Listener connection' },
]

const VIBRATO_SLIDERS: SliderConfig[] = [
  { key: 'rate_min', label: 'Rate Min', min: 3, max: 8, step: 0.1, unit: 'Hz', description: 'Minimum vibrato rate' },
  { key: 'rate_max', label: 'Rate Max', min: 3, max: 8, step: 0.1, unit: 'Hz', description: 'Maximum vibrato rate' },
  { key: 'onset_delay_min', label: 'Onset Min', min: 50, max: 500, step: 10, unit: 'ms', description: 'Min onset delay' },
  { key: 'onset_delay_max', label: 'Onset Max', min: 50, max: 500, step: 10, unit: 'ms', description: 'Max onset delay' },
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
// Components
// =============================================================================

function ParameterSlider({
  config,
  value,
  onChange,
  disabled = false,
}: {
  config: SliderConfig
  value: number
  onChange: (value: number) => void
  disabled?: boolean
}) {
  const displayValue = config.unit
    ? `${value.toFixed(config.step < 1 ? 1 : 0)}${config.unit}`
    : `${(value * 100).toFixed(0)}%`

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <label className="text-zinc-300" title={config.description}>
          {config.label}
        </label>
        <span className="text-zinc-500 font-mono">{displayValue}</span>
      </div>
      <input
        type="range"
        min={config.min}
        max={config.max}
        step={config.step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        className="w-full h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer
                   disabled:opacity-50 disabled:cursor-not-allowed
                   [&::-webkit-slider-thumb]:appearance-none
                   [&::-webkit-slider-thumb]:w-4
                   [&::-webkit-slider-thumb]:h-4
                   [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-purple-500
                   [&::-webkit-slider-thumb]:cursor-pointer
                   [&::-webkit-slider-thumb]:hover:bg-purple-400"
      />
    </div>
  )
}

function VoiceCard({
  voice,
  isSelected,
  onSelect,
}: {
  voice: Voice
  isSelected: boolean
  onSelect: () => void
}) {
  const isCustom = voice.name.startsWith('custom-') || voice.character.includes('Custom')

  return (
    <button
      onClick={onSelect}
      className={`w-full p-4 rounded-xl border transition-all text-left
        ${isSelected
          ? 'border-purple-500 bg-purple-500/10'
          : 'border-zinc-700 bg-zinc-800/50 hover:border-zinc-600'
        }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center
            ${isSelected ? 'bg-purple-500' : 'bg-zinc-700'}`}>
            {isCustom ? <Sliders className="w-5 h-5" /> : <User className="w-5 h-5" />}
          </div>
          <div>
            <h3 className="font-semibold text-white">{voice.name}</h3>
            <p className="text-sm text-zinc-400">{voice.classification}</p>
          </div>
        </div>
        {isSelected && (
          <div className="w-2 h-2 rounded-full bg-purple-500" />
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <span className="px-2 py-0.5 bg-zinc-700 rounded text-xs text-zinc-300">
          {midiToNote(voice.range_low)} - {midiToNote(voice.range_high)}
        </span>
        {voice.character.split(', ').map((trait) => (
          <span key={trait} className="px-2 py-0.5 bg-zinc-700/50 rounded text-xs text-zinc-400">
            {trait}
          </span>
        ))}
      </div>
    </button>
  )
}

function ParameterSection({
  title,
  icon: Icon,
  children,
  defaultOpen = true,
}: {
  title: string
  icon: React.ElementType
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="border border-zinc-700 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between bg-zinc-800/50 hover:bg-zinc-800"
      >
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-purple-400" />
          <span className="font-medium">{title}</span>
        </div>
        {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      {isOpen && (
        <div className="p-4 space-y-4 bg-zinc-900/50">
          {children}
        </div>
      )}
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

  // Edited parameters (for fine-tuning)
  const [editedTimbre, setEditedTimbre] = useState<VoiceTimbreParams | null>(null)
  const [editedEmotion, setEditedEmotion] = useState<VoiceEmotionParams | null>(null)
  const [editedVibrato, setEditedVibrato] = useState<VoiceVibratoParams | null>(null)

  // Fetch voices
  const { data: voicesData, isLoading, error } = useQuery({
    queryKey: ['voices'],
    queryFn: () => aetherApi.listVoices(),
  })

  // Preview mutation
  const previewMutation = useMutation({
    mutationFn: (name: string) => aetherApi.previewVoice(name),
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
    previewMutation.mutate(voice.name)
  }, [previewMutation])

  // Reset edits to original
  const resetEdits = useCallback(() => {
    if (selectedVoice) {
      setEditedTimbre({ ...selectedVoice.timbre })
      setEditedEmotion({ ...selectedVoice.emotion })
      setEditedVibrato({ ...selectedVoice.vibrato })
    }
  }, [selectedVoice])

  // Save as custom voice
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

  // Check if parameters have been modified
  const hasChanges = selectedVoice && editedTimbre && (
    JSON.stringify(editedTimbre) !== JSON.stringify(selectedVoice.timbre) ||
    JSON.stringify(editedEmotion) !== JSON.stringify(selectedVoice.emotion) ||
    JSON.stringify(editedVibrato) !== JSON.stringify(selectedVoice.vibrato)
  )

  if (isLoading) {
    return (
      <div className="min-h-screen bg-zinc-900 flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
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
      <header className="border-b border-zinc-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Mic className="w-6 h-6 text-purple-500" />
            <h1 className="text-xl font-semibold">Voice Manager</h1>
          </div>
          <span className="text-sm text-zinc-500">{voices.length} voices available</span>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Voice List */}
          <div className="lg:col-span-1 space-y-4">
            <h2 className="text-lg font-semibold text-zinc-300">Available Voices</h2>
            <div className="space-y-3">
              {voices.map((voice) => (
                <VoiceCard
                  key={voice.name}
                  voice={voice}
                  isSelected={selectedVoice?.name === voice.name}
                  onSelect={() => handleSelectVoice(voice)}
                />
              ))}
            </div>
          </div>

          {/* Voice Details & Fine-tuning */}
          <div className="lg:col-span-2 space-y-6">
            {selectedVoice ? (
              <>
                {/* Voice Info */}
                <div className="bg-zinc-800/50 rounded-xl p-6 border border-zinc-700">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h2 className="text-2xl font-bold">{selectedVoice.name}</h2>
                      <p className="text-zinc-400">{selectedVoice.classification}</p>
                    </div>
                    {selectedVoice.name.startsWith('custom-') && (
                      <button
                        onClick={() => deleteCustomMutation.mutate(selectedVoice.name)}
                        className="p-2 text-red-400 hover:bg-red-400/10 rounded-lg"
                        title="Delete custom voice"
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                    )}
                  </div>

                  {/* Preview */}
                  {previewMutation.data && (
                    <div className="bg-zinc-900 rounded-lg p-4 mb-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Volume2 className="w-4 h-4 text-purple-400" />
                        <span className="text-sm font-medium">Voice Description</span>
                      </div>
                      <p className="text-zinc-300 text-sm">{previewMutation.data.description}</p>
                    </div>
                  )}

                  {/* Range info */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-zinc-500">Comfortable Range</span>
                      <p className="font-mono">
                        {midiToNote(selectedVoice.range_low)} - {midiToNote(selectedVoice.range_high)}
                      </p>
                    </div>
                    <div>
                      <span className="text-zinc-500">Tessitura</span>
                      <p className="font-mono">
                        {midiToNote(selectedVoice.tessitura_low)} - {midiToNote(selectedVoice.tessitura_high)}
                      </p>
                    </div>
                  </div>
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
                        className="text-sm text-zinc-400 hover:text-white"
                      >
                        Reset to original
                      </button>
                    )}
                  </div>

                  {/* Timbre */}
                  <ParameterSection title="Timbre" icon={Volume2}>
                    {editedTimbre && TIMBRE_SLIDERS.map((config) => (
                      <ParameterSlider
                        key={config.key}
                        config={config}
                        value={editedTimbre[config.key as keyof VoiceTimbreParams]}
                        onChange={(v) => setEditedTimbre({ ...editedTimbre, [config.key]: v })}
                      />
                    ))}
                  </ParameterSection>

                  {/* Emotion */}
                  <ParameterSection title="Emotional Baseline" icon={User} defaultOpen={false}>
                    {editedEmotion && EMOTION_SLIDERS.map((config) => (
                      <ParameterSlider
                        key={config.key}
                        config={config}
                        value={editedEmotion[config.key as keyof VoiceEmotionParams]}
                        onChange={(v) => setEditedEmotion({ ...editedEmotion, [config.key]: v })}
                      />
                    ))}
                  </ParameterSection>

                  {/* Vibrato */}
                  <ParameterSection title="Vibrato" icon={Mic} defaultOpen={false}>
                    {editedVibrato && VIBRATO_SLIDERS.map((config) => (
                      <ParameterSlider
                        key={config.key}
                        config={config}
                        value={editedVibrato[config.key as keyof VoiceVibratoParams]}
                        onChange={(v) => setEditedVibrato({ ...editedVibrato, [config.key]: v })}
                      />
                    ))}
                  </ParameterSection>
                </div>

                {/* Save as Custom */}
                {hasChanges && (
                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-xl p-4">
                    {isCreatingCustom ? (
                      <div className="space-y-3">
                        <input
                          type="text"
                          value={customName}
                          onChange={(e) => setCustomName(e.target.value)}
                          placeholder="Custom voice name..."
                          className="w-full px-4 py-2 bg-zinc-800 border border-zinc-600 rounded-lg
                                     focus:outline-none focus:border-purple-500"
                        />
                        <div className="flex gap-2">
                          <button
                            onClick={handleSaveCustom}
                            disabled={!customName.trim() || createCustomMutation.isPending}
                            className="flex-1 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg
                                       disabled:opacity-50 disabled:cursor-not-allowed
                                       flex items-center justify-center gap-2"
                          >
                            {createCustomMutation.isPending ? (
                              <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                              <Save className="w-4 h-4" />
                            )}
                            Save
                          </button>
                          <button
                            onClick={() => setIsCreatingCustom(false)}
                            className="px-4 py-2 bg-zinc-700 hover:bg-zinc-600 rounded-lg"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <button
                        onClick={() => setIsCreatingCustom(true)}
                        className="w-full py-2 flex items-center justify-center gap-2
                                   text-purple-400 hover:text-purple-300"
                      >
                        <Plus className="w-4 h-4" />
                        Save as Custom Voice
                      </button>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="bg-zinc-800/30 rounded-xl p-12 border border-dashed border-zinc-700
                              flex flex-col items-center justify-center text-center">
                <Mic className="w-12 h-12 text-zinc-600 mb-4" />
                <h3 className="text-lg font-medium text-zinc-400 mb-2">Select a Voice</h3>
                <p className="text-zinc-500 text-sm max-w-md">
                  Choose a voice from the list to view its details and fine-tune parameters.
                  You can create custom voice presets by adjusting the parameters.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
