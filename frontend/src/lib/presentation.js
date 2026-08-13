const AGENT_LABELS = {
  general: '服务协调 Agent',
  technical: '技术可靠性 Agent',
  billing: '收入与合规 Agent',
  escalation: '人工升级通道',
  system: '系统'
}

const INTENT_LABELS = {
  query: '信息查询',
  complaint: '投诉反馈',
  request: '业务办理',
  greeting: '问候',
  escalation: '转人工',
  technical: '技术问题',
  billing: '账单退款',
  account: '账户管理',
  feedback: '正向反馈',
  other: '其他'
}

export function agentLabel(value) {
  return AGENT_LABELS[value] || value || '未分配'
}

export function intentLabel(value) {
  return INTENT_LABELS[value] || value || '待识别'
}

export function formatLatency(value) {
  const number = Number(value)
  if (!Number.isFinite(number) || number <= 0) return '—'
  return number >= 1000 ? `${(number / 1000).toFixed(1)} s` : `${Math.round(number)} ms`
}

export function formatPercent(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  return `${Math.round(number * 100)}%`
}

export function friendlyError(error) {
  const status = Number(error?.status || 0)
  if (status === 400 || status === 422) return '提交内容格式不正确，请检查后重试。'
  if (status === 401 || status === 403) return '当前请求没有访问权限，请检查服务配置。'
  if (status === 413) return '文件超过 10 MB 限制，请压缩或拆分后重试。'
  if (status === 502 || status === 503 || status === 504) {
    return '后端服务暂时不可用，请确认服务已启动后重试。'
  }
  if (/failed to fetch|networkerror|network request failed/i.test(error?.message || '')) {
    return '无法连接后端服务，请检查 API 地址和网络设置。'
  }
  return error?.message || '操作失败，请稍后重试。'
}

export function safeJson(value) {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value ?? '')
  }
}

export function collectionEntries(value) {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? Object.entries(value)
    : []
}
