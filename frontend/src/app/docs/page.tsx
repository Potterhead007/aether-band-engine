'use client'

import { useState } from 'react'
import { Book, Code, Zap, Server, ChevronDown, ChevronRight, Copy, Check } from 'lucide-react'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://aether-api-production.up.railway.app'

interface EndpointProps {
  method: 'GET' | 'POST'
  path: string
  description: string
  requestBody?: string
  responseBody: string
}

function Endpoint({ method, path, description, requestBody, responseBody }: EndpointProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [copied, setCopied] = useState(false)

  const methodColors = {
    GET: 'bg-green-500/20 text-green-400 border-green-500/30',
    POST: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  }

  const curlCommand = method === 'GET'
    ? `curl ${API_BASE}${path}`
    : `curl -X POST ${API_BASE}${path} \\
  -H "Content-Type: application/json" \\
  -d '${requestBody || '{}'}'`

  const handleCopy = () => {
    navigator.clipboard.writeText(curlCommand)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="border border-slate-700/50 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-3 p-4 hover:bg-slate-800/50 transition-colors text-left"
      >
        <span className={`px-2 py-1 rounded text-xs font-mono border ${methodColors[method]}`}>
          {method}
        </span>
        <code className="text-slate-300 font-mono text-sm flex-1">{path}</code>
        <span className="text-slate-400 text-sm hidden md:block">{description}</span>
        {isOpen ? (
          <ChevronDown className="w-4 h-4 text-slate-500" />
        ) : (
          <ChevronRight className="w-4 h-4 text-slate-500" />
        )}
      </button>

      {isOpen && (
        <div className="p-4 border-t border-slate-700/50 bg-slate-900/50 space-y-4">
          <p className="text-slate-400 text-sm md:hidden">{description}</p>

          {/* cURL Example */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-slate-500 uppercase tracking-wider">cURL Example</span>
              <button
                onClick={handleCopy}
                className="text-xs text-slate-400 hover:text-white flex items-center gap-1"
              >
                {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                {copied ? 'Copied!' : 'Copy'}
              </button>
            </div>
            <pre className="bg-slate-950 rounded-lg p-3 overflow-x-auto text-sm text-slate-300">
              {curlCommand}
            </pre>
          </div>

          {/* Request Body */}
          {requestBody && (
            <div>
              <span className="text-xs text-slate-500 uppercase tracking-wider block mb-2">
                Request Body
              </span>
              <pre className="bg-slate-950 rounded-lg p-3 overflow-x-auto text-sm text-green-400">
                {requestBody}
              </pre>
            </div>
          )}

          {/* Response */}
          <div>
            <span className="text-xs text-slate-500 uppercase tracking-wider block mb-2">
              Response
            </span>
            <pre className="bg-slate-950 rounded-lg p-3 overflow-x-auto text-sm text-blue-400">
              {responseBody}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}

export default function DocsPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      {/* Header */}
      <h1 className="text-3xl font-bold mb-2">
        <span className="gradient-text">API</span> Documentation
      </h1>
      <p className="text-slate-400 mb-8">
        Integrate AETHER music generation into your applications.
      </p>

      {/* Base URL */}
      <div className="card mb-8">
        <div className="flex items-center gap-2 mb-2">
          <Server className="w-4 h-4 text-aether-400" />
          <span className="text-sm font-medium text-slate-300">Base URL</span>
        </div>
        <code className="text-aether-400 font-mono bg-slate-900/50 px-3 py-2 rounded-lg block">
          {API_BASE}
        </code>
      </div>

      {/* Quick Start */}
      <section className="mb-12">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-400" />
          Quick Start
        </h2>
        <div className="card">
          <p className="text-slate-400 mb-4">
            Generate a track with a single API call:
          </p>
          <pre className="bg-slate-950 rounded-lg p-4 overflow-x-auto text-sm text-slate-300">
{`curl -X POST ${API_BASE}/v1/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "title": "My Track",
    "genre": "synthwave",
    "brief": "An energetic synthwave track with arpeggios"
  }'`}
          </pre>
        </div>
      </section>

      {/* Endpoints */}
      <section className="mb-12">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Code className="w-5 h-5 text-blue-400" />
          Endpoints
        </h2>

        <div className="space-y-3">
          <Endpoint
            method="GET"
            path="/health"
            description="Check API health status"
            responseBody={`{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "components": {
    "health": true,
    "metrics": true,
    "providers": true
  }
}`}
          />

          <Endpoint
            method="GET"
            path="/v1/genres"
            description="List available music genres"
            responseBody={`{
  "genres": [
    { "id": "synthwave", "name": "Synthwave" },
    { "id": "lo-fi-hip-hop", "name": "Lo-Fi Hip Hop" },
    { "id": "jazz", "name": "Jazz" },
    { "id": "techno", "name": "Techno" },
    ...
  ]
}`}
          />

          <Endpoint
            method="POST"
            path="/v1/generate"
            description="Generate a new music track"
            requestBody={`{
  "title": "Neon Dreams",
  "genre": "synthwave",
  "brief": "An energetic track with arpeggios and pads",
  "bpm": 118,
  "key": "Am",
  "duration_seconds": 180
}`}
            responseBody={`{
  "job_id": "abc123...",
  "status": "completed",
  "song_spec": { ... },
  "harmony_spec": { ... },
  "melody_spec": { ... },
  "arrangement_spec": { ... }
}`}
          />

          <Endpoint
            method="POST"
            path="/v1/render"
            description="Render audio from specs (local only)"
            requestBody={`{
  "song_spec": { ... },
  "harmony_spec": { ... },
  "melody_spec": { ... },
  "arrangement_spec": { ... },
  "output_formats": ["mp3", "wav"]
}`}
            responseBody={`{
  "job_id": "xyz789...",
  "status": "completed",
  "duration_seconds": 180,
  "output_files": {
    "mp3": "/output/track.mp3",
    "wav": "/output/track.wav"
  }
}`}
          />
        </div>
      </section>

      {/* Genres Reference */}
      <section className="mb-12">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Book className="w-5 h-5 text-purple-400" />
          Available Genres
        </h2>
        <div className="card">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {[
              'synthwave', 'lo-fi-hip-hop', 'jazz', 'techno', 'house', 'ambient',
              'rock', 'funk', 'disco', 'trap', 'drum-and-bass', 'dubstep',
              'r-and-b', 'neo-soul', 'chillwave', 'acoustic-folk', 'cinematic', 'pop'
            ].map((genre) => (
              <code key={genre} className="text-sm text-slate-300 bg-slate-900/50 px-2 py-1 rounded">
                {genre}
              </code>
            ))}
          </div>
        </div>
      </section>

      {/* Rate Limits */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Rate Limits</h2>
        <div className="card">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400">
                <th className="pb-2">Endpoint</th>
                <th className="pb-2">Limit</th>
              </tr>
            </thead>
            <tbody className="text-slate-300">
              <tr>
                <td className="py-2 font-mono">/v1/generate</td>
                <td>10 requests/minute</td>
              </tr>
              <tr>
                <td className="py-2 font-mono">/v1/render</td>
                <td>5 requests/minute</td>
              </tr>
              <tr>
                <td className="py-2 font-mono">Other endpoints</td>
                <td>60 requests/minute</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
