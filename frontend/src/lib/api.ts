/**
 * AETHER API Client
 *
 * Production-grade API client with:
 * - Request timeouts with AbortController
 * - Automatic retry with exponential backoff
 * - Request ID correlation for distributed tracing
 * - Comprehensive error handling
 * - TypeScript strict mode compliance
 */

// =============================================================================
// Configuration
// =============================================================================

const API_BASE_URL = ((): string => {
  const url = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const trimmed = url.trim()

  // Debug logging - always log in browser
  if (typeof window !== 'undefined') {
    console.info('[AETHER] API URL:', trimmed)
    console.info('[AETHER] ENV:', process.env.NODE_ENV)
    // Also show in an alert for debugging
    if (!sessionStorage.getItem('aether_url_shown')) {
      sessionStorage.setItem('aether_url_shown', 'true')
      alert('AETHER API URL: ' + trimmed)
    }
  }

  if (typeof window !== 'undefined' && process.env.NODE_ENV === 'production') {
    if (!trimmed.startsWith('https://')) {
      console.error('[AETHER] SECURITY WARNING: Production API URL should use HTTPS')
    }
  }

  return trimmed
})()

const CONFIG = {
  timeouts: {
    default: 30_000,      // 30 seconds
    generation: 300_000,  // 5 minutes
    render: 600_000,      // 10 minutes
  },
  retry: {
    maxAttempts: 3,
    baseDelayMs: 1000,
    maxDelayMs: 10_000,
  },
} as const

// =============================================================================
// Types
// =============================================================================

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

export interface ApiErrorDetails {
  type?: string
  title?: string
  status?: number
  detail?: string
  instance?: string
  request_id?: string
}

// =============================================================================
// Error Classes
// =============================================================================

export class ApiError extends Error {
  public readonly status: number
  public readonly requestId: string
  public readonly details: ApiErrorDetails

  constructor(message: string, status: number, requestId: string, details: ApiErrorDetails = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.requestId = requestId
    this.details = details
    Object.setPrototypeOf(this, ApiError.prototype)
  }

  get isRetryable(): boolean {
    return this.status >= 500 || this.status === 429
  }
}

export class TimeoutError extends Error {
  public readonly requestId: string

  constructor(requestId: string, timeoutMs: number) {
    super(`Request timed out after ${timeoutMs}ms`)
    this.name = 'TimeoutError'
    this.requestId = requestId
    Object.setPrototypeOf(this, TimeoutError.prototype)
  }
}

export class NetworkError extends Error {
  public readonly requestId: string
  public readonly cause?: Error

  constructor(requestId: string, cause?: Error) {
    super(cause?.message || 'Network request failed')
    this.name = 'NetworkError'
    this.requestId = requestId
    this.cause = cause
    Object.setPrototypeOf(this, NetworkError.prototype)
  }
}

// =============================================================================
// Utilities
// =============================================================================

function generateRequestId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 11)}`
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function calculateBackoff(attempt: number): number {
  const delay = CONFIG.retry.baseDelayMs * Math.pow(2, attempt)
  const jitter = Math.random() * 0.3 * delay
  return Math.min(delay + jitter, CONFIG.retry.maxDelayMs)
}

// =============================================================================
// HTTP Client
// =============================================================================

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
  body?: unknown
  timeout?: number
  retry?: boolean
}

async function request<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const {
    method = 'GET',
    body,
    timeout = CONFIG.timeouts.default,
    retry = true,
  } = options

  const requestId = generateRequestId()
  const url = `${API_BASE_URL}${endpoint}`

  const headers: Record<string, string> = {
    'Accept': 'application/json',
    'X-Request-ID': requestId,
  }

  if (body) {
    headers['Content-Type'] = 'application/json'
  }

  const fetchOptions: RequestInit = {
    method,
    headers,
    mode: 'cors',
    body: body ? JSON.stringify(body) : undefined,
  }

  let lastError: Error | null = null
  const maxAttempts = retry ? CONFIG.retry.maxAttempts : 1

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeout)

    try {
      if (attempt > 0) {
        const backoff = calculateBackoff(attempt - 1)
        console.debug(`[AETHER] Retry attempt ${attempt + 1}/${maxAttempts} after ${backoff}ms`)
        await sleep(backoff)
      }

      console.debug(`[AETHER] Fetching: ${method} ${url}`)

      const response = await fetch(url, {
        ...fetchOptions,
        signal: controller.signal,
      })

      console.debug(`[AETHER] Response: ${response.status} ${response.statusText}`)

      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorDetails = await parseErrorResponse(response)
        const error = new ApiError(
          errorDetails.detail || errorDetails.title || `HTTP ${response.status}`,
          response.status,
          requestId,
          errorDetails
        )

        if (error.isRetryable && attempt < maxAttempts - 1) {
          lastError = error
          continue
        }

        throw error
      }

      const data = await response.json()
      return data as T

    } catch (err) {
      clearTimeout(timeoutId)

      if (err instanceof ApiError) {
        throw err
      }

      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          throw new TimeoutError(requestId, timeout)
        }

        lastError = new NetworkError(requestId, err)

        if (attempt < maxAttempts - 1) {
          continue
        }
      }

      throw lastError || new NetworkError(requestId)
    }
  }

  throw lastError || new NetworkError(requestId)
}

async function parseErrorResponse(response: Response): Promise<ApiErrorDetails> {
  try {
    const text = await response.text()
    if (!text) {
      return { status: response.status, title: response.statusText }
    }

    const json = JSON.parse(text)
    return {
      type: json.type,
      title: json.title || json.error,
      status: json.status || response.status,
      detail: json.detail || json.message,
      instance: json.instance,
      request_id: json.request_id,
    }
  } catch {
    return { status: response.status, title: response.statusText }
  }
}

// =============================================================================
// API Methods
// =============================================================================

export const aetherApi = {
  /**
   * Health check endpoint
   */
  async health(): Promise<HealthResponse> {
    return request<HealthResponse>('/health')
  },

  /**
   * List available genres
   */
  async listGenres(): Promise<{ genres: Genre[] }> {
    return request<{ genres: Genre[] }>('/v1/genres')
  },

  /**
   * Generate a new music track
   */
  async generate(params: GenerateRequest): Promise<GenerateResponse> {
    console.info(`[AETHER] Starting generation: "${params.title}"`)

    const result = await request<GenerateResponse>('/v1/generate', {
      method: 'POST',
      body: params,
      timeout: CONFIG.timeouts.generation,
      retry: false, // Don't retry generation to avoid duplicates
    })

    console.info(`[AETHER] Generation complete: ${result.job_id}`)
    return result
  },

  /**
   * Render audio from specifications
   */
  async render(params: RenderRequest): Promise<RenderResponse> {
    console.info('[AETHER] Starting render')

    const result = await request<RenderResponse>('/v1/render', {
      method: 'POST',
      body: params,
      timeout: CONFIG.timeouts.render,
      retry: false, // Don't retry render to avoid duplicates
    })

    console.info(`[AETHER] Render complete: ${result.job_id}`)
    return result
  },
}

// =============================================================================
// Exports
// =============================================================================

// Default export for convenience
export default aetherApi

// Named export for explicit imports
export { aetherApi as api }
