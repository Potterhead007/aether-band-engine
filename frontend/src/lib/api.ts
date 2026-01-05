const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface GenerateRequest {
  title: string
  genre: string
  brief: string
  bpm?: number
  key?: string
  duration_seconds?: number
}

export interface GenerateResponse {
  job_id: string
  status: string
  song_spec?: Record<string, unknown>
  harmony_spec?: Record<string, unknown>
  melody_spec?: Record<string, unknown>
  arrangement_spec?: Record<string, unknown>
}

export interface RenderRequest {
  song_spec: Record<string, unknown>
  harmony_spec?: Record<string, unknown>
  melody_spec?: Record<string, unknown>
  arrangement_spec?: Record<string, unknown>
  output_formats?: string[]
  render_stems?: boolean
}

export interface RenderResponse {
  job_id: string
  status: string
  duration_seconds: number
  loudness_lufs?: number
  peak_db?: number
  output_files: Record<string, string>
}

export interface HealthResponse {
  status: string
  version: string
  uptime_seconds: number
  components: Record<string, boolean>
}

export interface Genre {
  id: string
  name: string
  aliases: string[]
}

class AetherAPI {
  private baseUrl: string

  constructor(baseUrl: string = API_URL) {
    this.baseUrl = baseUrl
  }

  async health(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health`)
    if (!response.ok) throw new Error('Health check failed')
    return response.json()
  }

  async listGenres(): Promise<{ genres: Genre[] }> {
    const response = await fetch(`${this.baseUrl}/v1/genres`)
    if (!response.ok) throw new Error('Failed to fetch genres')
    return response.json()
  }

  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await fetch(`${this.baseUrl}/v1/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    if (!response.ok) {
      const text = await response.text()
      let detail = 'Generation failed'
      try {
        const error = JSON.parse(text)
        detail = error.detail || detail
      } catch {
        if (text) detail = text
      }
      throw new Error(detail)
    }
    return response.json()
  }

  async render(request: RenderRequest): Promise<RenderResponse> {
    const response = await fetch(`${this.baseUrl}/v1/render`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    if (!response.ok) {
      const text = await response.text()
      let detail = 'Rendering failed'
      try {
        const error = JSON.parse(text)
        detail = error.detail || detail
      } catch {
        if (text) detail = text
      }
      throw new Error(detail)
    }
    return response.json()
  }
}

export const api = new AetherAPI()
export default api
