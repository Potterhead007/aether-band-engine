'use client'

import { useState, useCallback } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Loader2, Music, CheckCircle, AlertTriangle, RefreshCw } from 'lucide-react'
import {
  aetherApi,
  GenerateRequest,
  GenerateResponse,
  RenderRequest,
  RenderResponse,
  ApiError,
  TimeoutError,
  NetworkError,
} from '@/lib/api'

// =============================================================================
// Types
// =============================================================================

type GenerationStep = 'idle' | 'generating' | 'rendering' | 'complete' | 'error'

interface FormData {
  title: string
  genre: string
  brief: string
  bpm: number
  key: string
  duration_seconds: number
}

const INITIAL_FORM_DATA: FormData = {
  title: '',
  genre: 'synthwave',
  brief: '',
  bpm: 120,
  key: 'Am',
  duration_seconds: 180,
}

const MUSICAL_KEYS = [
  'C', 'Cm', 'D', 'Dm', 'E', 'Em', 'F', 'Fm', 'G', 'Gm', 'A', 'Am', 'B', 'Bm'
] as const

// =============================================================================
// Error Formatting
// =============================================================================

function formatError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 429) {
      return 'Rate limit exceeded. Please wait a moment and try again.'
    }
    if (error.status === 401) {
      return 'Authentication required. Please check your API configuration.'
    }
    if (error.status >= 500) {
      return 'Server error. Our team has been notified. Please try again later.'
    }
    return error.message
  }

  if (error instanceof TimeoutError) {
    return 'Request timed out. The server may be busy. Please try again.'
  }

  if (error instanceof NetworkError) {
    return 'Network error. Please check your internet connection and try again.'
  }

  if (error instanceof Error) {
    return error.message
  }

  return 'An unexpected error occurred. Please try again.'
}

// =============================================================================
// Component
// =============================================================================

export default function GeneratePage() {
  // State
  const [step, setStep] = useState<GenerationStep>('idle')
  const [formData, setFormData] = useState<FormData>(INITIAL_FORM_DATA)
  const [generateResult, setGenerateResult] = useState<GenerateResponse | null>(null)
  const [renderResult, setRenderResult] = useState<RenderResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Queries
  const genresQuery = useQuery({
    queryKey: ['genres'],
    queryFn: () => aetherApi.listGenres(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 2,
  })

  // Mutations
  const generateMutation = useMutation({
    mutationFn: (data: GenerateRequest) => aetherApi.generate(data),
    onSuccess: (data) => {
      setGenerateResult(data)
      // Skip render for now - show generation result directly
      setRenderResult({
        job_id: data.job_id,
        status: 'completed',
        duration_seconds: 180,
        output_files: {},
      })
      setStep('complete')
    },
    onError: (err: unknown) => {
      setError(formatError(err))
      setStep('error')
    },
  })

  const renderMutation = useMutation({
    mutationFn: (data: RenderRequest) => aetherApi.render(data),
    onSuccess: (data) => {
      setRenderResult(data)
      setStep('complete')
    },
    onError: (err: unknown) => {
      setError(formatError(err))
      setStep('error')
    },
  })

  // Handlers
  const triggerRender = useCallback((genData: GenerateResponse) => {
    const renderRequest: RenderRequest = {
      song_spec: genData.song_spec || {},
      harmony_spec: genData.harmony_spec,
      melody_spec: genData.melody_spec,
      arrangement_spec: genData.arrangement_spec,
      output_formats: ['wav', 'mp3'],
      render_stems: false,
    }
    renderMutation.mutate(renderRequest)
  }, [renderMutation])

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()

    // Validate
    if (!formData.title.trim()) {
      setError('Please enter a track title')
      return
    }
    if (!formData.brief.trim()) {
      setError('Please enter a creative brief')
      return
    }

    setError(null)
    setStep('generating')
    generateMutation.mutate(formData)
  }, [formData, generateMutation])

  const handleReset = useCallback(() => {
    setStep('idle')
    setGenerateResult(null)
    setRenderResult(null)
    setError(null)
    generateMutation.reset()
    renderMutation.reset()
  }, [generateMutation, renderMutation])

  const handleFieldChange = useCallback(<K extends keyof FormData>(
    field: K,
    value: FormData[K]
  ) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    if (error) setError(null)
  }, [error])

  // Computed
  const isLoading = step === 'generating' || step === 'rendering'
  const stepIndex = ['idle', 'generating', 'rendering', 'complete'].indexOf(
    step === 'error' ? 'idle' : step
  )

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      {/* Header */}
      <h1 className="text-3xl font-bold mb-2">
        <span className="gradient-text">Generate</span> Music
      </h1>
      <p className="text-slate-400 mb-8">
        Describe your track and let AETHER compose, arrange, and render it.
      </p>

      {/* Progress Steps */}
      <ProgressSteps currentStep={stepIndex} isError={step === 'error'} />

      {/* Error State */}
      {step === 'error' && error && (
        <ErrorCard message={error} onRetry={handleReset} />
      )}

      {/* Inline Error */}
      {step === 'idle' && error && (
        <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Form */}
      {step === 'idle' && (
        <GenerateForm
          formData={formData}
          genres={genresQuery.data?.genres || []}
          genresLoading={genresQuery.isLoading}
          onFieldChange={handleFieldChange}
          onSubmit={handleSubmit}
        />
      )}

      {/* Loading State */}
      {isLoading && (
        <LoadingCard step={step as 'generating' | 'rendering'} />
      )}

      {/* Complete State */}
      {step === 'complete' && renderResult && (
        <CompleteCard
          title={formData.title}
          renderResult={renderResult}
          generateResult={generateResult}
          onReset={handleReset}
        />
      )}
    </div>
  )
}

// =============================================================================
// Sub-components
// =============================================================================

interface ProgressStepsProps {
  currentStep: number
  isError: boolean
}

function ProgressSteps({ currentStep, isError }: ProgressStepsProps) {
  const steps = ['Configure', 'Generate', 'Render', 'Complete']

  return (
    <div className="flex items-center justify-between mb-8">
      {steps.map((label, i) => {
        const isActive = i <= currentStep && !isError
        const isCurrent = i === currentStep && !isError
        const isComplete = i < currentStep

        return (
          <div key={label} className="flex items-center">
            <div className={`
              w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors
              ${isActive ? 'bg-aether-600 text-white' : 'bg-slate-700 text-slate-400'}
              ${isCurrent ? 'ring-2 ring-aether-400 ring-offset-2 ring-offset-slate-900' : ''}
              ${isError && i === 0 ? 'bg-red-600' : ''}
            `}>
              {isComplete ? (
                <CheckCircle className="w-5 h-5" />
              ) : isError && i === 0 ? (
                <AlertTriangle className="w-4 h-4" />
              ) : (
                i + 1
              )}
            </div>
            <span className={`ml-2 text-sm ${isActive ? 'text-white' : 'text-slate-500'}`}>
              {label}
            </span>
            {i < 3 && (
              <div className={`w-16 h-0.5 mx-4 transition-colors ${
                isComplete ? 'bg-aether-600' : 'bg-slate-700'
              }`} />
            )}
          </div>
        )
      })}
    </div>
  )
}

interface ErrorCardProps {
  message: string
  onRetry: () => void
}

function ErrorCard({ message, onRetry }: ErrorCardProps) {
  return (
    <div className="card border-red-500/50 mb-6">
      <div className="flex items-start gap-3 mb-4">
        <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
        <div className="text-red-400">{message}</div>
      </div>
      <button onClick={onRetry} className="btn-secondary inline-flex items-center gap-2">
        <RefreshCw className="w-4 h-4" />
        Try Again
      </button>
    </div>
  )
}

interface GenerateFormProps {
  formData: FormData
  genres: Array<{ id: string; name: string }>
  genresLoading: boolean
  onFieldChange: <K extends keyof FormData>(field: K, value: FormData[K]) => void
  onSubmit: (e: React.FormEvent) => void
}

function GenerateForm({
  formData,
  genres,
  genresLoading,
  onFieldChange,
  onSubmit,
}: GenerateFormProps) {
  return (
    <form onSubmit={onSubmit} className="card space-y-6">
      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <label htmlFor="title" className="label">Track Title</label>
          <input
            id="title"
            type="text"
            className="input"
            placeholder="Enter a title..."
            value={formData.title}
            onChange={(e) => onFieldChange('title', e.target.value)}
            maxLength={100}
            required
          />
        </div>

        <div>
          <label htmlFor="genre" className="label">Genre</label>
          <select
            id="genre"
            className="input"
            value={formData.genre}
            onChange={(e) => onFieldChange('genre', e.target.value)}
            disabled={genresLoading}
          >
            {genresLoading ? (
              <option>Loading genres...</option>
            ) : genres.length > 0 ? (
              genres.map((genre) => (
                <option key={genre.id} value={genre.id}>
                  {genre.name}
                </option>
              ))
            ) : (
              <option value="synthwave">Synthwave</option>
            )}
          </select>
        </div>
      </div>

      <div>
        <label htmlFor="brief" className="label">Creative Brief</label>
        <textarea
          id="brief"
          className="input min-h-[120px]"
          placeholder="Describe the mood, style, and feel you want..."
          value={formData.brief}
          onChange={(e) => onFieldChange('brief', e.target.value)}
          maxLength={2000}
          required
        />
        <div className="text-xs text-slate-500 mt-1 text-right">
          {formData.brief.length}/2000
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        <div>
          <label htmlFor="bpm" className="label">BPM</label>
          <input
            id="bpm"
            type="number"
            className="input"
            min={40}
            max={300}
            value={formData.bpm}
            onChange={(e) => onFieldChange('bpm', parseInt(e.target.value) || 120)}
          />
        </div>

        <div>
          <label htmlFor="key" className="label">Key</label>
          <select
            id="key"
            className="input"
            value={formData.key}
            onChange={(e) => onFieldChange('key', e.target.value)}
          >
            {MUSICAL_KEYS.map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="duration" className="label">Duration (seconds)</label>
          <input
            id="duration"
            type="number"
            className="input"
            min={30}
            max={600}
            value={formData.duration_seconds}
            onChange={(e) => onFieldChange('duration_seconds', parseInt(e.target.value) || 180)}
          />
        </div>
      </div>

      <button type="submit" className="btn-primary w-full py-3 text-lg">
        <Music className="w-5 h-5 inline mr-2" />
        Generate Track
      </button>
    </form>
  )
}

interface LoadingCardProps {
  step: 'generating' | 'rendering'
}

function LoadingCard({ step }: LoadingCardProps) {
  const isGenerating = step === 'generating'

  return (
    <div className="card text-center py-12">
      <Loader2 className="w-12 h-12 text-aether-400 animate-spin mx-auto mb-4" />
      <h3 className="text-xl font-semibold mb-2">
        {isGenerating ? 'Generating Music...' : 'Rendering Audio...'}
      </h3>
      <p className="text-slate-400">
        {isGenerating
          ? 'AI is composing your track based on your creative brief.'
          : 'Converting your composition to high-quality audio files.'}
      </p>
      <p className="text-slate-500 text-sm mt-4">
        {isGenerating ? 'This may take up to 5 minutes.' : 'This may take up to 10 minutes.'}
      </p>
    </div>
  )
}

interface CompleteCardProps {
  title: string
  renderResult: RenderResponse
  generateResult: GenerateResponse | null
  onReset: () => void
}

function CompleteCard({ title, renderResult, generateResult, onReset }: CompleteCardProps) {
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <>
      <div className="card">
        <div className="text-center mb-8">
          <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center mx-auto mb-4">
            <CheckCircle className="w-8 h-8 text-green-400" />
          </div>
          <h3 className="text-2xl font-semibold mb-2">{title}</h3>
          <p className="text-slate-400">Your track has been generated successfully!</p>
        </div>

        <div className="grid md:grid-cols-3 gap-4 mb-8">
          <div className="bg-slate-900/50 rounded-lg p-4 text-center">
            <div className="text-sm text-slate-400 mb-1">Duration</div>
            <div className="text-lg font-mono">
              {formatDuration(renderResult.duration_seconds)}
            </div>
          </div>
          <div className="bg-slate-900/50 rounded-lg p-4 text-center">
            <div className="text-sm text-slate-400 mb-1">Loudness</div>
            <div className="text-lg font-mono">
              {renderResult.loudness_lufs?.toFixed(1) || '-14.0'} LUFS
            </div>
          </div>
          <div className="bg-slate-900/50 rounded-lg p-4 text-center">
            <div className="text-sm text-slate-400 mb-1">Files</div>
            <div className="text-lg font-mono">
              {Object.keys(renderResult.output_files).length}
            </div>
          </div>
        </div>

        <div className="space-y-3 mb-8">
          <h4 className="font-medium text-slate-300">Output Files</h4>
          {Object.entries(renderResult.output_files).map(([key, path]) => (
            <div key={key} className="flex items-center justify-between bg-slate-900/50 rounded-lg p-3">
              <span className="text-sm text-slate-400">{key}</span>
              <span className="text-sm font-mono text-slate-300 truncate max-w-md">{path}</span>
            </div>
          ))}
        </div>

        <button onClick={onReset} className="btn-secondary w-full">
          Create Another Track
        </button>
      </div>

      {/* Song Spec Preview */}
      {generateResult?.song_spec && (
        <details className="card mt-6">
          <summary className="cursor-pointer font-medium text-slate-300 hover:text-white">
            View Song Specification
          </summary>
          <pre className="mt-4 p-4 bg-slate-900/50 rounded-lg overflow-x-auto text-sm text-slate-400">
            {JSON.stringify(generateResult.song_spec, null, 2)}
          </pre>
        </details>
      )}
    </>
  )
}
