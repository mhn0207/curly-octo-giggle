<template>
  <section class="feature-workspace" aria-labelledby="monitor-title">
    <header class="feature-hero monitor-hero">
      <div>
        <span class="section-kicker">LIVE OBSERVABILITY</span>
        <h2 id="monitor-title">看见每个 Agent 的真实表现</h2>
        <p>成功率、延迟、路由评分与工具熔断状态，共同形成在线优化闭环。</p>
      </div>
      <button class="button button-primary" type="button" :disabled="loading" @click="$emit('refresh')">
        <span v-if="loading" class="spinner" aria-hidden="true"></span>
        {{ loading ? '正在刷新' : '刷新数据' }}
      </button>
    </header>

    <p v-if="error" class="inline-alert error" role="alert">{{ error }}</p>

    <div class="metric-grid">
      <article>
        <span class="metric-icon blue" aria-hidden="true">A</span>
        <div><small>在线 Agent</small><strong>{{ agentEntries.length }}</strong></div>
        <em>动态路由</em>
      </article>
      <article>
        <span class="metric-icon cyan" aria-hidden="true">T</span>
        <div><small>受控工具</small><strong>{{ toolCount }}</strong></div>
        <em>权限治理</em>
      </article>
      <article>
        <span class="metric-icon violet" aria-hidden="true">K</span>
        <div><small>知识片段</small><strong>{{ knowledgeCount }}</strong></div>
        <em>RAG 增强</em>
      </article>
      <article>
        <span class="metric-icon amber" aria-hidden="true">!</span>
        <div><small>活跃告警</small><strong>{{ alerts.length }}</strong></div>
        <em>{{ alerts.length ? '需要关注' : '运行平稳' }}</em>
      </article>
    </div>

    <div v-if="agentEntries.length || toolEntries.length" class="monitor-grid">
      <section class="data-card">
        <div class="panel-title-row">
          <div>
            <span class="section-kicker">ROUTING HEALTH</span>
            <h3>Agent 在线表现</h3>
          </div>
          <span class="data-count">{{ agentEntries.length }} AGENTS</span>
        </div>
        <div class="table-scroll">
          <table>
            <thead>
              <tr><th>Agent</th><th>请求</th><th>成功率</th><th>平均耗时</th><th>路由评分</th></tr>
            </thead>
            <tbody>
              <tr v-for="[name, stats] in agentEntries" :key="name">
                <td><span :class="['agent-dot', name]"></span><strong>{{ agentLabel(name) }}</strong></td>
                <td>{{ stats.total ?? 0 }}</td>
                <td><span :class="['table-status', rateTone(stats.success_rate)]">{{ formatPercent(stats.success_rate) }}</span></td>
                <td>{{ formatLatency(stats.avg_ms ?? stats.avg_latency_ms) }}</td>
                <td>{{ decimal(stats.routing_score) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="data-card">
        <div class="panel-title-row">
          <div>
            <span class="section-kicker">TOOL GOVERNANCE</span>
            <h3>工具运行状态</h3>
          </div>
          <span class="data-count">{{ toolCount }} TOOLS</span>
        </div>
        <div class="table-scroll">
          <table>
            <thead>
              <tr><th>工具</th><th>调用</th><th>成功率</th><th>P95 耗时</th><th>熔断器</th></tr>
            </thead>
            <tbody>
              <tr v-for="[name, stats] in toolEntries" :key="name">
                <td><strong>{{ name }}</strong></td>
                <td>{{ stats.total ?? 0 }}</td>
                <td><span :class="['table-status', rateTone(stats.success_rate)]">{{ formatPercent(stats.success_rate) }}</span></td>
                <td>{{ formatLatency(stats.p95_latency_ms ?? stats.avg_latency_ms) }}</td>
                <td><span :class="['table-status', stats.circuit_state === 'open' ? 'bad' : 'good']">{{ stats.circuit_state || 'closed' }}</span></td>
              </tr>
              <tr v-if="!toolEntries.length">
                <td colspan="5" class="empty-cell">还没有工具调用统计</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>

    <div v-else-if="!loading" class="monitor-empty">
      <span aria-hidden="true">⌁</span>
      <h3>尚未取得运行数据</h3>
      <p>启动后端服务并刷新，即可查看 Agent 路由和工具健康度。</p>
      <button class="button button-primary" type="button" @click="$emit('refresh')">重新连接</button>
    </div>

    <div class="insight-grid">
      <section class="data-card insight-card">
        <div class="panel-title-row">
          <div>
            <span class="section-kicker">ACTIVE ALERTS</span>
            <h3>告警与异常</h3>
          </div>
          <span :class="['data-count', alerts.length ? 'danger' : 'success']">{{ alerts.length || 'CLEAR' }}</span>
        </div>
        <ul v-if="alerts.length" class="insight-list">
          <li v-for="(alert, index) in alerts" :key="alert.id || index">
            <span :class="['insight-mark', alert.severity || 'warning']" aria-hidden="true">!</span>
            <span><strong>{{ alert.metric || '运行告警' }}</strong><small>{{ alert.message }}</small></span>
          </li>
        </ul>
        <div v-else class="compact-empty"><span>✓</span>当前没有活跃告警</div>
      </section>

      <section class="data-card insight-card">
        <div class="panel-title-row">
          <div>
            <span class="section-kicker">OPTIMIZATION LOOP</span>
            <h3>路由优化建议</h3>
          </div>
          <span class="data-count">{{ suggestions.length }}</span>
        </div>
        <ul v-if="suggestions.length" class="insight-list">
          <li v-for="(item, index) in suggestions" :key="item.title || index">
            <span class="insight-mark info" aria-hidden="true">↗</span>
            <span><strong>{{ item.title }}</strong><small>{{ item.action }}</small></span>
          </li>
        </ul>
        <div v-else class="compact-empty"><span>◇</span>暂无需要执行的优化建议</div>
      </section>
    </div>

    <section v-if="toolCatalog.length || recentExecutions.length" class="data-card governance-card">
      <div class="panel-title-row">
        <div>
          <span class="section-kicker">AUDIT TRAIL</span>
          <h3>工具目录与最近执行</h3>
        </div>
        <span class="data-count">{{ skillCount }} SKILLS</span>
      </div>
      <div class="governance-grid">
        <div>
          <h4>已注册工具</h4>
          <ul class="catalog-list">
            <li v-for="tool in toolCatalog" :key="tool.name || tool.tool_name">
              <span><strong>{{ tool.name || tool.tool_name }}</strong><small>{{ tool.description || '受控业务工具' }}</small></span>
              <span :class="['risk-badge', tool.risk_level || tool.risk || 'low']">{{ tool.risk_level || tool.risk || 'low' }}</span>
            </li>
          </ul>
        </div>
        <div>
          <h4>最近执行记录</h4>
          <ul class="execution-list">
            <li v-for="(item, index) in recentExecutions.slice().reverse().slice(0, 8)" :key="item.execution_id || index">
              <span :class="['execution-state', item.success === false ? 'error' : 'success']"></span>
              <span><strong>{{ item.tool_name }}</strong><small>{{ item.agent_type || 'system' }} · {{ formatLatency(item.latency_ms) }}</small></span>
              <time>{{ formatClock(item.timestamp) }}</time>
            </li>
          </ul>
        </div>
      </div>
    </section>

    <details v-if="monitor?.structured_output || monitor?.rag" class="diagnostic-details">
      <summary>查看结构化输出与 RAG 诊断数据</summary>
      <pre>{{ safeJson({ structured_output: monitor?.structured_output, rag: monitor?.rag }) }}</pre>
    </details>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { agentLabel, collectionEntries, formatLatency, formatPercent, safeJson } from '../lib/presentation'

const props = defineProps({
  monitor: { type: Object, default: null },
  knowledgeCount: { type: [String, Number], default: '—' },
  toolCatalog: { type: Array, default: () => [] },
  recentExecutions: { type: Array, default: () => [] },
  skillsSummary: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' }
})

defineEmits(['refresh'])

const agentEntries = computed(() => collectionEntries(props.monitor?.agent_stats))
const toolEntries = computed(() => collectionEntries(props.monitor?.tool_stats))
const alerts = computed(() => props.monitor?.active_alerts || [])
const suggestions = computed(() => props.monitor?.suggestions || [])
const toolCount = computed(() => props.toolCatalog.length || toolEntries.value.length)
const skillCount = computed(() => {
  const value = props.skillsSummary
  if (!value) return 0
  if (Array.isArray(value.skills)) return value.skills.length
  if (Array.isArray(value.loaded_skills)) return value.loaded_skills.length
  return value.total ?? value.count ?? Object.keys(value.skills || {}).length ?? 0
})

function rateTone(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return 'neutral'
  if (number >= 0.95) return 'good'
  if (number >= 0.85) return 'warning'
  return 'bad'
}

function decimal(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number.toFixed(3) : '—'
}

function formatClock(value) {
  if (!value) return ''
  return new Intl.DateTimeFormat('zh-CN', {
    hour: '2-digit',
    minute: '2-digit'
  }).format(new Date(value))
}
</script>
