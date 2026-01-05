'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import Link from 'next/link'
import { Activity, Music, Zap, Layers } from 'lucide-react'

function StatusBadge({ status }: { status: string }) {
  const colors = {
    healthy: 'bg-green-500/20 text-green-400 border-green-500/30',
    degraded: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    unhealthy: 'bg-red-500/20 text-red-400 border-red-500/30',
  }
  const color = colors[status as keyof typeof colors] || colors.unhealthy

  return (
    <span className={`px-3 py-1 rounded-full text-sm border ${color}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  )
}

function WaveformAnimation() {
  return (
    <div className="flex items-end justify-center h-16 space-x-1">
      {[...Array(5)].map((_, i) => (
        <div
          key={i}
          className="waveform-bar w-2 bg-gradient-to-t from-aether-600 to-aether-400 rounded-full"
          style={{ height: '100%' }}
        />
      ))}
    </div>
  )
}

export default function Home() {
  const { data: health, isLoading } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.health(),
    refetchInterval: 30000,
  })

  const { data: genresData } = useQuery({
    queryKey: ['genres'],
    queryFn: () => api.listGenres(),
  })

  return (
    <div className="max-w-7xl mx-auto px-4 py-12">
      {/* Hero Section */}
      <section className="text-center mb-16">
        <div className="mb-8">
          <WaveformAnimation />
        </div>
        <h1 className="text-5xl font-bold mb-4">
          <span className="gradient-text">AETHER</span> Band Engine
        </h1>
        <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-8">
          Autonomous Ensemble for Thoughtful Harmonic Expression and Rendering.
          Generate professional-quality music with AI.
        </p>
        <div className="flex justify-center gap-4">
          <Link href="/generate" className="btn-primary text-lg px-8 py-3">
            Start Creating
          </Link>
          <a href="/docs" className="btn-secondary text-lg px-8 py-3">
            API Documentation
          </a>
        </div>
      </section>

      {/* Status Card */}
      <section className="mb-16">
        <div className="card">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <Activity className="w-5 h-5 text-aether-400" />
              System Status
            </h2>
            {health && <StatusBadge status={health.status} />}
          </div>

          {isLoading ? (
            <div className="text-slate-400">Checking system health...</div>
          ) : health ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-900/50 rounded-lg p-4">
                <div className="text-sm text-slate-400 mb-1">Version</div>
                <div className="font-mono text-lg">{health.version}</div>
              </div>
              <div className="bg-slate-900/50 rounded-lg p-4">
                <div className="text-sm text-slate-400 mb-1">Uptime</div>
                <div className="font-mono text-lg">
                  {Math.floor(health.uptime_seconds / 3600)}h {Math.floor((health.uptime_seconds % 3600) / 60)}m
                </div>
              </div>
              <div className="bg-slate-900/50 rounded-lg p-4">
                <div className="text-sm text-slate-400 mb-1">Components</div>
                <div className="font-mono text-lg">
                  {Object.values(health.components).filter(Boolean).length}/{Object.keys(health.components).length}
                </div>
              </div>
              <div className="bg-slate-900/50 rounded-lg p-4">
                <div className="text-sm text-slate-400 mb-1">Genres</div>
                <div className="font-mono text-lg">{genresData?.genres.length || 0}</div>
              </div>
            </div>
          ) : (
            <div className="text-red-400">Unable to connect to API</div>
          )}
        </div>
      </section>

      {/* Features */}
      <section className="mb-16">
        <h2 className="text-2xl font-bold text-center mb-8">Capabilities</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="card text-center">
            <div className="w-12 h-12 rounded-full bg-aether-500/20 flex items-center justify-center mx-auto mb-4">
              <Music className="w-6 h-6 text-aether-400" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Multi-Genre Music</h3>
            <p className="text-slate-400 text-sm">
              Generate music across {genresData?.genres.length || 10}+ genres including synthwave, lo-fi, electronic, and more.
            </p>
          </div>

          <div className="card text-center">
            <div className="w-12 h-12 rounded-full bg-purple-500/20 flex items-center justify-center mx-auto mb-4">
              <Zap className="w-6 h-6 text-purple-400" />
            </div>
            <h3 className="text-lg font-semibold mb-2">AI-Powered</h3>
            <p className="text-slate-400 text-sm">
              Advanced LLM integration for creative direction, composition, and arrangement.
            </p>
          </div>

          <div className="card text-center">
            <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center mx-auto mb-4">
              <Layers className="w-6 h-6 text-green-400" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Production Ready</h3>
            <p className="text-slate-400 text-sm">
              Export mastered WAV and MP3 files with ITU-R BS.1770-4 loudness compliance.
            </p>
          </div>
        </div>
      </section>

      {/* Available Genres */}
      {genresData && (
        <section>
          <h2 className="text-2xl font-bold text-center mb-8">Available Genres</h2>
          <div className="flex flex-wrap justify-center gap-2">
            {genresData.genres.map((genre) => (
              <span
                key={genre.id}
                className="px-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-full text-sm text-slate-300 hover:bg-slate-700/50 transition-colors cursor-default"
              >
                {genre.name}
              </span>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
