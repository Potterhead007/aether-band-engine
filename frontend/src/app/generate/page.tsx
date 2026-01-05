'use client'

import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { api, GenerateResponse, RenderResponse } from '@/lib/api'
import { Loader2, Music, Download, Play, CheckCircle } from 'lucide-react'

type GenerationStep = 'idle' | 'generating' | 'rendering' | 'complete' | 'error'

export default function GeneratePage() {
  const [step, setStep] = useState<GenerationStep>('idle')
  const [formData, setFormData] = useState({
    title: '',
    genre: 'synthwave',
    brief: '',
    bpm: 120,
    key: 'Am',
    duration_seconds: 180,
  })
  const [generateResult, setGenerateResult] = useState<GenerateResponse | null>(null)
  const [renderResult, setRenderResult] = useState<RenderResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const { data: genresData } = useQuery({
    queryKey: ['genres'],
    queryFn: () => api.listGenres(),
  })

  const generateMutation = useMutation({
    mutationFn: api.generate,
    onSuccess: (data) => {
      setGenerateResult(data)
      setStep('rendering')
      // Auto-trigger render
      renderMutation.mutate({
        song_spec: data.song_spec || {},
        harmony_spec: data.harmony_spec,
        melody_spec: data.melody_spec,
        arrangement_spec: data.arrangement_spec,
        output_formats: ['wav', 'mp3'],
        render_stems: false,
      })
    },
    onError: (err: Error) => {
      setError(err.message)
      setStep('error')
    },
  })

  const renderMutation = useMutation({
    mutationFn: api.render,
    onSuccess: (data) => {
      setRenderResult(data)
      setStep('complete')
    },
    onError: (err: Error) => {
      setError(err.message)
      setStep('error')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setStep('generating')
    generateMutation.mutate(formData)
  }

  const handleReset = () => {
    setStep('idle')
    setGenerateResult(null)
    setRenderResult(null)
    setError(null)
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <h1 className="text-3xl font-bold mb-2">
        <span className="gradient-text">Generate</span> Music
      </h1>
      <p className="text-slate-400 mb-8">
        Describe your track and let AETHER compose, arrange, and render it.
      </p>

      {/* Progress Steps */}
      <div className="flex items-center justify-between mb-8">
        {['Configure', 'Generate', 'Render', 'Complete'].map((label, i) => {
          const stepStates = ['idle', 'generating', 'rendering', 'complete']
          const currentIdx = stepStates.indexOf(step)
          const isActive = i <= currentIdx
          const isCurrent = i === currentIdx

          return (
            <div key={label} className="flex items-center">
              <div className={`
                w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium
                ${isActive ? 'bg-aether-600 text-white' : 'bg-slate-700 text-slate-400'}
                ${isCurrent && step !== 'complete' && step !== 'error' ? 'ring-2 ring-aether-400 ring-offset-2 ring-offset-slate-900' : ''}
              `}>
                {i < currentIdx || step === 'complete' ? (
                  <CheckCircle className="w-5 h-5" />
                ) : (
                  i + 1
                )}
              </div>
              <span className={`ml-2 text-sm ${isActive ? 'text-white' : 'text-slate-500'}`}>
                {label}
              </span>
              {i < 3 && (
                <div className={`w-16 h-0.5 mx-4 ${isActive && i < currentIdx ? 'bg-aether-600' : 'bg-slate-700'}`} />
              )}
            </div>
          )
        })}
      </div>

      {/* Error State */}
      {step === 'error' && (
        <div className="card border-red-500/50 mb-6">
          <div className="text-red-400 mb-4">{error}</div>
          <button onClick={handleReset} className="btn-secondary">
            Try Again
          </button>
        </div>
      )}

      {/* Form */}
      {step === 'idle' && (
        <form onSubmit={handleSubmit} className="card space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="label">Track Title</label>
              <input
                type="text"
                className="input"
                placeholder="Enter a title..."
                value={formData.title}
                onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                required
              />
            </div>

            <div>
              <label className="label">Genre</label>
              <select
                className="input"
                value={formData.genre}
                onChange={(e) => setFormData({ ...formData, genre: e.target.value })}
              >
                {genresData?.genres.map((genre) => (
                  <option key={genre.id} value={genre.id}>
                    {genre.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="label">Creative Brief</label>
            <textarea
              className="input min-h-[120px]"
              placeholder="Describe the mood, style, and feel you want..."
              value={formData.brief}
              onChange={(e) => setFormData({ ...formData, brief: e.target.value })}
              required
            />
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <label className="label">BPM</label>
              <input
                type="number"
                className="input"
                min={40}
                max={300}
                value={formData.bpm}
                onChange={(e) => setFormData({ ...formData, bpm: parseInt(e.target.value) })}
              />
            </div>

            <div>
              <label className="label">Key</label>
              <select
                className="input"
                value={formData.key}
                onChange={(e) => setFormData({ ...formData, key: e.target.value })}
              >
                {['C', 'Cm', 'D', 'Dm', 'E', 'Em', 'F', 'Fm', 'G', 'Gm', 'A', 'Am', 'B', 'Bm'].map((k) => (
                  <option key={k} value={k}>{k}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="label">Duration (seconds)</label>
              <input
                type="number"
                className="input"
                min={30}
                max={600}
                value={formData.duration_seconds}
                onChange={(e) => setFormData({ ...formData, duration_seconds: parseInt(e.target.value) })}
              />
            </div>
          </div>

          <button type="submit" className="btn-primary w-full py-3 text-lg">
            <Music className="w-5 h-5 inline mr-2" />
            Generate Track
          </button>
        </form>
      )}

      {/* Generation Progress */}
      {(step === 'generating' || step === 'rendering') && (
        <div className="card text-center py-12">
          <Loader2 className="w-12 h-12 text-aether-400 animate-spin mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">
            {step === 'generating' ? 'Generating Music...' : 'Rendering Audio...'}
          </h3>
          <p className="text-slate-400">
            {step === 'generating'
              ? 'AI is composing your track based on your creative brief.'
              : 'Converting your composition to high-quality audio files.'}
          </p>
        </div>
      )}

      {/* Complete State */}
      {step === 'complete' && renderResult && (
        <div className="card">
          <div className="text-center mb-8">
            <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center mx-auto mb-4">
              <CheckCircle className="w-8 h-8 text-green-400" />
            </div>
            <h3 className="text-2xl font-semibold mb-2">{formData.title}</h3>
            <p className="text-slate-400">Your track has been generated successfully!</p>
          </div>

          <div className="grid md:grid-cols-3 gap-4 mb-8">
            <div className="bg-slate-900/50 rounded-lg p-4 text-center">
              <div className="text-sm text-slate-400 mb-1">Duration</div>
              <div className="text-lg font-mono">
                {Math.floor(renderResult.duration_seconds / 60)}:{String(Math.floor(renderResult.duration_seconds % 60)).padStart(2, '0')}
              </div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-4 text-center">
              <div className="text-sm text-slate-400 mb-1">Loudness</div>
              <div className="text-lg font-mono">{renderResult.loudness_lufs?.toFixed(1) || '-14.0'} LUFS</div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-4 text-center">
              <div className="text-sm text-slate-400 mb-1">Files</div>
              <div className="text-lg font-mono">{Object.keys(renderResult.output_files).length}</div>
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

          <div className="flex gap-4">
            <button onClick={handleReset} className="btn-secondary flex-1">
              Create Another
            </button>
          </div>
        </div>
      )}

      {/* Song Spec Preview */}
      {generateResult?.song_spec && step === 'complete' && (
        <details className="card mt-6">
          <summary className="cursor-pointer font-medium text-slate-300 hover:text-white">
            View Song Specification
          </summary>
          <pre className="mt-4 p-4 bg-slate-900/50 rounded-lg overflow-x-auto text-sm text-slate-400">
            {JSON.stringify(generateResult.song_spec, null, 2)}
          </pre>
        </details>
      )}
    </div>
  )
}
