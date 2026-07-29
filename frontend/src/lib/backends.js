const DEFAULT_BACKEND = {
  id: 'zhiying',
  label: '知应 AI',
  baseUrl: import.meta.env.VITE_ZHIYING_API_URL || '/api/zhiying',
  port: '8000'
}

export function createInitialSettings() {
  const saved = readSettings()
  return {
    userId: saved.userId || 'u1001',
    conversationId: saved.conversationId || '',
    endpoints: {
      zhiying: saved.endpoints?.zhiying || DEFAULT_BACKEND.baseUrl
    }
  }
}

export function saveSettings(settings) {
  localStorage.setItem('zhiying.frontend.settings', JSON.stringify(settings))
}

export function backendMeta(settings) {
  return {
    ...DEFAULT_BACKEND,
    baseUrl: normalizeBaseUrl(settings.endpoints.zhiying || DEFAULT_BACKEND.baseUrl)
  }
}

export async function requestHealth(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/health')
}

export async function requestMonitor(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/monitor')
}

export async function requestKnowledgeStats(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/knowledge/stats')
}

export async function requestSearch(settings, query, topK = 5) {
  const params = new URLSearchParams({ query, top_k: String(topK) })
  return requestJson(backendMeta(settings).baseUrl, `/search?${params}`, { method: 'POST' })
}

export async function requestChat(settings, message) {
  const raw = await requestJson(backendMeta(settings).baseUrl, '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      user_id: settings.userId || 'anonymous',
      conv_id: settings.conversationId || undefined
    })
  })
  return normalizeChatResponse(raw)
}

export async function addKnowledge(settings, documents) {
  return requestJson(backendMeta(settings).baseUrl, '/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents })
  })
}

export async function uploadKnowledge(settings, file) {
  const form = new FormData()
  form.append('file', file)
  return requestJson(backendMeta(settings).baseUrl, '/knowledge/upload', {
    method: 'POST',
    body: form
  })
}

function normalizeChatResponse(raw) {
  return {
    backend: DEFAULT_BACKEND.id,
    conversationId: raw.conv_id || raw.conversation_id || raw.conversationId || '',
    requestId: raw.request_id || raw.requestId || '',
    response: raw.response || '',
    intent: raw.intent || 'other',
    agentType: raw.agent_type || raw.agentType || '',
    escalated: Boolean(raw.escalated),
    latencyMs: Number(raw.latency_ms ?? raw.latencyMs ?? 0),
    knowledgeUsed: Boolean(raw.knowledge_used ?? raw.knowledgeUsed),
    toolCalls: Array.isArray(raw.tool_calls) ? raw.tool_calls : [],
    synthesis: raw.synthesis && typeof raw.synthesis === 'object' ? raw.synthesis : null,
    raw
  }
}

async function requestJson(baseUrl, path, options = {}) {
  const url = `${normalizeBaseUrl(baseUrl)}${path}`
  const response = await fetch(url, options)
  const text = await response.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!response.ok) {
    const detail = typeof data === 'string' ? data : JSON.stringify(data)
    throw new Error(`${response.status} ${response.statusText}: ${detail}`)
  }
  return data
}

function normalizeBaseUrl(value) {
  return String(value || '').replace(/\/+$/, '')
}

function readSettings() {
  try {
    return JSON.parse(localStorage.getItem('zhiying.frontend.settings') || '{}')
  } catch {
    return {}
  }
}