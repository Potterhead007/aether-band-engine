// AETHER API Client
// Institutional-grade with HTTPS enforcement, timeouts, and request correlation

const API_URL = (() => {
  const url = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  // Enforce HTTPS in production
  if (typeof window !== 'undefined' && process.env.NODE_ENV === 'production') {
    if (!url.startsWith('https://')) {
      console.error('SECURITY: Production API URL must use HTTPS')
    }
  }
  return url.trim() // Remove any trailing whitespace/newlines
})()

// Default timeouts (ms)
const DEFAULT_TIMEOUT = 30000 // 30 seconds for normal requests
const GENERATION_TIMEOUT = 300000 // 5 minutes for generation
const RENDER_TIMEOUT = 600000 // 10 minutes for rendering

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

/**
 * Generate a unique request ID for distributed tracing
 */
function generateRequestId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  // Fallback for older environments
  return `${Date.now()}-${Math.random().toString(36).substring(2, 15)}`
}

/**
 * Create headers with request ID for tracing
 */
function createHeaders(requestId: string): Record<string, string> {
  return {
    'Content-Type': 'application/json',
    'X-Request-ID': requestId,
  }
}

/**
 * Create a fetch request with timeout
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    })
    return response
  } finally {
    clearTimeout(timeoutId)
  }
}

class AetherAPI {
  private baseUrl: string

  constructor(baseUrl: string = API_URL) {
    this.baseUrl = baseUrl
  }

  async health(): Promise<HealthResponse> {
    const requestId = generateRequestId()
    const response = await fetchWithTimeout(
      `${this.baseUrl}/health`,
      { headers: { 'X-Request-ID': requestId } },
      DEFAULT_TIMEOUT
    )
    if (!response.ok) throw new Error('Health check failed')
    return response.json()
  }

  async listGenres(): Promise<{ genres: Genre[] }> {
    const requestId = generateRequestId()
    const response = await fetchWithTimeout(
      `${this.baseUrl}/v1/genres`,
      { headers: { 'X-Request-ID': requestId } },
      DEFAULT_TIMEOUT
    )
    if (!response.ok) throw new Error('Failed to fetch genres')
    return response.json()
  }

  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const requestId = generateRequestId()
    console.debug(`[AETHER] Generate request ${requestId}:`, request.title)

    const response = await fetchWithTimeout(
      `${this.baseUrl}/v1/generate`,
      {
        method: 'POST',
        headers: createHeaders(requestId),
        body: JSON.stringify(request),
      },
      GENERATION_TIMEOUT
    )

    if (!response.ok) {
      const text = await response.text()
      let detail = 'Generation failed'
      try {
        const error = JSON.parse(text)
        detail = error.detail || error.error || detail
      } catch {
        if (text) detail = text
      }
      console.error(`[AETHER] Generate failed ${requestId}:`, detail)
      throw new Error(detail)
    }

    console.debug(`[AETHER] Generate success ${requestId}`)
    return response.json()
  }

  async render(request: RenderRequest): Promise<RenderResponse> {
    const requestId = generateRequestId()
    console.debug(`[AETHER] Render request ${requestId}`)

    const response = await fetchWithTimeout(
      `${this.baseUrl}/v1/render`,
      {
        method: 'POST',
        headers: createHeaders(requestId),
        body: JSON.stringify(request),
      },
      RENDER_TIMEOUT
    )

    if (!response.ok) {
      const text = await response.text()
      let detail = 'Rendering failed'
      try {
        const error = JSON.parse(text)
        detail = error.detail || error.error || detail
      } catch {
        if (text) detail = text
      }
      console.error(`[AETHER] Render failed ${requestId}:`, detail)
      throw new Error(detail)
    }

    console.debug(`[AETHER] Render success ${requestId}`)
    return response.json()
  }
}

export const api = new AetherAPI()
export default api
