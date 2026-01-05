'use client'

import { useState } from 'react'
import { Music, Clock, Disc, ChevronRight } from 'lucide-react'
import Link from 'next/link'

interface Track {
  id: string
  title: string
  genre: string
  bpm: number
  key: string
  duration: string
  createdAt: string
  status: 'completed' | 'processing' | 'failed'
}

// Demo tracks - in production this would come from an API
const DEMO_TRACKS: Track[] = [
  {
    id: '1',
    title: 'Neon Dreams',
    genre: 'Synthwave',
    bpm: 118,
    key: 'Am',
    duration: '3:24',
    createdAt: '2 hours ago',
    status: 'completed',
  },
  {
    id: '2',
    title: 'Midnight Drive',
    genre: 'Lo-Fi Hip Hop',
    bpm: 85,
    key: 'Dm',
    duration: '2:58',
    createdAt: '5 hours ago',
    status: 'completed',
  },
  {
    id: '3',
    title: 'Electric Pulse',
    genre: 'Techno',
    bpm: 130,
    key: 'Cm',
    duration: '4:12',
    createdAt: '1 day ago',
    status: 'completed',
  },
]

function StatusBadge({ status }: { status: Track['status'] }) {
  const styles = {
    completed: 'bg-green-500/20 text-green-400 border-green-500/30',
    processing: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    failed: 'bg-red-500/20 text-red-400 border-red-500/30',
  }

  return (
    <span className={`px-2 py-0.5 rounded text-xs border ${styles[status]}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  )
}

function TrackCard({ track }: { track: Track }) {
  return (
    <div className="card hover:border-aether-500/50 transition-colors group cursor-pointer">
      <div className="flex items-center gap-4">
        <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-aether-600 to-purple-600 flex items-center justify-center flex-shrink-0">
          <Disc className="w-8 h-8 text-white" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-semibold text-white truncate">{track.title}</h3>
            <StatusBadge status={track.status} />
          </div>
          <p className="text-slate-400 text-sm">{track.genre}</p>
          <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
            <span>{track.bpm} BPM</span>
            <span>{track.key}</span>
            <span>{track.duration}</span>
          </div>
        </div>

        <div className="text-right flex-shrink-0">
          <div className="text-slate-500 text-sm flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {track.createdAt}
          </div>
          <ChevronRight className="w-5 h-5 text-slate-600 group-hover:text-aether-400 mt-2 ml-auto transition-colors" />
        </div>
      </div>
    </div>
  )
}

export default function HistoryPage() {
  const [tracks] = useState<Track[]>(DEMO_TRACKS)

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">
            <span className="gradient-text">Generation</span> History
          </h1>
          <p className="text-slate-400">
            View and manage your previously generated tracks.
          </p>
        </div>
        <Link href="/generate" className="btn-primary">
          <Music className="w-4 h-4 mr-2 inline" />
          New Track
        </Link>
      </div>

      {/* Info Banner */}
      <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mb-8">
        <p className="text-blue-400 text-sm">
          <strong>Coming Soon:</strong> Track history persistence requires user accounts.
          Currently showing demo tracks. Generate a new track to see it here during your session.
        </p>
      </div>

      {/* Track List */}
      {tracks.length > 0 ? (
        <div className="space-y-4">
          {tracks.map((track) => (
            <TrackCard key={track.id} track={track} />
          ))}
        </div>
      ) : (
        <div className="card text-center py-16">
          <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
            <Music className="w-8 h-8 text-slate-600" />
          </div>
          <h3 className="text-xl font-semibold mb-2">No tracks yet</h3>
          <p className="text-slate-400 mb-6">
            Generate your first track to see it here.
          </p>
          <Link href="/generate" className="btn-primary inline-flex items-center">
            <Music className="w-4 h-4 mr-2" />
            Generate Track
          </Link>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mt-8">
        <div className="card text-center py-6">
          <div className="text-3xl font-bold text-aether-400 mb-1">{tracks.length}</div>
          <div className="text-slate-400 text-sm">Tracks Generated</div>
        </div>
        <div className="card text-center py-6">
          <div className="text-3xl font-bold text-purple-400 mb-1">3</div>
          <div className="text-slate-400 text-sm">Genres Used</div>
        </div>
        <div className="card text-center py-6">
          <div className="text-3xl font-bold text-green-400 mb-1">10:34</div>
          <div className="text-slate-400 text-sm">Total Duration</div>
        </div>
      </div>
    </div>
  )
}
