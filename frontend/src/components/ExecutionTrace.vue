<template>
  <aside class="trace-panel" aria-labelledby="trace-title">
    <div class="panel-title-row">
      <div>
        <span class="section-kicker">ORCHESTRATION</span>
        <h2 id="trace-title">执行轨迹</h2>
      </div>
      <span v-if="trace?.requestId" class="request-id">#{{ trace.requestId }}</span>
    </div>

    <ol class="pipeline" aria-label="多 Agent 执行流程">
      <li v-for="(step, index) in steps" :key="step.label" :class="step.status">
        <span class="step-node" aria-hidden="true">
          <span v-if="step.status === 'active'" class="step-pulse"></span>
          <svg v-else-if="step.status === 'done'" viewBox="0 0 20 20"><path d="M5 10.5l3 3L15 6.5" /></svg>
          <span v-else>{{ index + 1 }}</span>
        </span>
        <span class="step-copy">
          <strong>{{ step.label }}</strong>
          <small>{{ step.detail }}</small>
        </span>
      </li>
    </ol>

    <div v-if="trace" class="trace-summary">
      <span><b>{{ formatLatency(trace.latencyMs) }}</b> 总耗时</span>
      <span><b>{{ toolCalls.length }}</b> 工具调用</span>
      <span><b>{{ agents.length }}</b> 参与 Agent</span>
    </div>

    <div v-if="flags.length" class="trace-flags" aria-label="处理标记">
      <span v-for="flag in flags" :key="flag.text" :class="flag.tone">{{ flag.text }}</span>
    </div>

    <section v-if="agents.length" class="trace-section">
      <h3>Agent 协作</h3>
      <ul class="agent-list">
        <li v-for="agent in agents" :key="agent.agent_type">
          <span :class="['agent-avatar', agent.agent_type]" aria-hidden="true">{{ agentInitial(agent.agent_type) }}</span>
          <span>
            <strong>{{ agentLabel(agent.agent_type) }}</strong>
            <small>{{ agent.fact_count || 0 }} 条事实 · {{ agent.tool_call_count || 0 }} 次工具</small>
          </span>
          <span :class="['compact-status', agent.success === false ? 'error' : 'success']">
            {{ agent.success === false ? '异常' : '完成' }}
          </span>
        </li>
      </ul>
    </section>

    <section v-if="facts.length" class="trace-section">
      <h3>已确认事实</h3>
      <dl class="fact-list">
        <div v-for="fact in facts" :key="fact.key || fact.label">
          <dt>{{ fact.label || fact.key }}</dt>
          <dd>{{ displayValue(fact.value) }}</dd>
        </div>
      </dl>
    </section>

    <section v-if="toolCalls.length" class="trace-section">
      <h3>工具调用</h3>
      <details v-for="(call, index) in toolCalls" :key="call.tool_use_id || call.tool_name + index" class="tool-call">
        <summary>
          <span class="tool-symbol" aria-hidden="true">↗</span>
          <span>
            <strong>{{ call.tool_name || 'unknown_tool' }}</strong>
            <small>{{ formatLatency(call.latency_ms) }}{{ call.cached ? ' · 缓存命中' : '' }}</small>
          </span>
          <span :class="['compact-status', call.success === false ? 'error' : 'success']">
            {{ call.success === false ? '失败' : '成功' }}
          </span>
        </summary>
        <pre>{{ safeJson(call.result ?? call.result_preview ?? call.error ?? '无返回数据') }}</pre>
      </details>
    </section>

    <section v-if="conflicts.length" class="trace-section">
      <h3>信息冲突</h3>
      <div v-for="conflict in conflicts" :key="conflict.key" class="conflict-card">
        <strong>{{ conflict.label || conflict.key }}</strong>
        <p>{{ conflict.resolution || '需要进一步确认' }}</p>
      </div>
    </section>

    <div v-if="!trace && !loading" class="trace-empty">
      <span aria-hidden="true">◇</span>
      <strong>等待一次对话</strong>
      <p>发送消息后，这里会展示从意图识别到结果综合的完整链路。</p>
    </div>
  </aside>
</template>

<script setup>
import { computed } from 'vue'
import { agentLabel, formatLatency, intentLabel, safeJson } from '../lib/presentation'

const props = defineProps({
  trace: { type: Object, default: null },
  loading: { type: Boolean, default: false }
})

const toolCalls = computed(() => props.trace?.toolCalls || [])
const facts = computed(() => props.trace?.synthesis?.confirmed_facts || [])
const conflicts = computed(() => props.trace?.synthesis?.conflicts || [])
const agents = computed(() => {
  if (props.trace?.synthesis?.agents?.length) return props.trace.synthesis.agents
  if (!props.trace?.agentType) return []
  return [{
    agent_type: props.trace.agentType,
    success: true,
    fact_count: facts.value.length,
    tool_call_count: toolCalls.value.length
  }]
})

const steps = computed(() => {
  if (!props.trace) {
    return [
      { label: '意图识别', detail: props.loading ? '正在理解用户诉求' : '等待输入', status: props.loading ? 'active' : 'pending' },
      { label: 'Agent 路由', detail: '选择最合适的处理角色', status: 'pending' },
      { label: '知识增强', detail: '按需召回 RAG 上下文', status: 'pending' },
      { label: '工具执行', detail: '受控调用业务工具', status: 'pending' },
      { label: '结果综合', detail: '合并证据并生成回复', status: 'pending' }
    ]
  }
  const synthesisCount = props.trace.synthesis?.agents?.length || 1
  return [
    { label: '意图识别', detail: intentLabel(props.trace.intent), status: 'done' },
    { label: 'Agent 路由', detail: agentLabel(props.trace.agentType), status: 'done' },
    {
      label: '知识增强',
      detail: props.trace.knowledgeUsed ? 'RAG 知识已命中' : '本轮无需知识库',
      status: props.trace.knowledgeUsed ? 'done' : 'skipped'
    },
    {
      label: '工具执行',
      detail: toolCalls.value.length ? String(toolCalls.value.length) + ' 次受控调用' : '本轮无需工具',
      status: toolCalls.value.length ? 'done' : 'skipped'
    },
    {
      label: '结果综合',
      detail: synthesisCount > 1 ? String(synthesisCount) + ' 个 Agent 结果已合并' : '单 Agent 完成回复',
      status: 'done'
    }
  ]
})

const flags = computed(() => {
  const values = []
  const synthesis = props.trace?.synthesis
  if (props.trace?.escalated || synthesis?.requires_human) values.push({ text: '需要人工跟进', tone: 'danger' })
  if (synthesis?.requires_approval) values.push({ text: '等待业务审批', tone: 'warning' })
  if (synthesis?.partial_failure) values.push({ text: '部分 Agent 失败', tone: 'warning' })
  if (props.trace?.knowledgeUsed) values.push({ text: '知识库增强', tone: 'info' })
  return values
})

function agentInitial(type) {
  return agentLabel(type).slice(0, 1)
}

function displayValue(value) {
  if (typeof value === 'object') return safeJson(value).replace(/\s+/g, ' ')
  if (typeof value === 'boolean') return value ? '是' : '否'
  return String(value ?? '—')
}
</script>
