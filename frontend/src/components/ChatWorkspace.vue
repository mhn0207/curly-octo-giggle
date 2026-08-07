<template>
  <div class="chat-workspace">
    <section class="chat-card" aria-labelledby="chat-title">
      <header class="chat-card-header">
        <div>
          <span class="section-kicker">LIVE CONVERSATION</span>
          <h2 id="chat-title">智能客服对话</h2>
          <p>{{ conversationId ? '会话 ' + conversationId : '新会话将在首次回复后自动创建' }}</p>
        </div>
        <button v-if="messages.length" class="button button-ghost button-small" type="button" @click="$emit('clear')">
          清空会话
        </button>
      </header>

      <div ref="messageList" class="message-list" aria-live="polite" :aria-busy="loading">
        <div v-if="!messages.length && !loading" class="conversation-empty">
          <span class="empty-orbit" aria-hidden="true"><i></i><i></i><b>AI</b></span>
          <h3>让多 Agent 协作处理一次真实问题</h3>
          <p>系统会自动识别意图、选择处理角色，并在需要时检索知识或调用业务工具。</p>
          <div class="prompt-suggestions" aria-label="示例问题">
            <button v-for="prompt in prompts" :key="prompt" type="button" @click="$emit('send', prompt)">
              {{ prompt }}
            </button>
          </div>
        </div>

        <article
          v-for="item in messages"
          :key="item.id"
          :class="['message-row', item.role, { failed: item.failed, selected: selectedMessageId === item.id }]"
        >
          <span v-if="item.role === 'assistant'" class="message-avatar" aria-hidden="true">知</span>
          <div class="message-content">
            <div class="message-meta">
              <strong>{{ item.role === 'user' ? '你' : agentLabel(item.trace?.agentType) }}</strong>
              <span v-if="item.trace?.intent">{{ intentLabel(item.trace.intent) }}</span>
              <time>{{ formatTime(item.createdAt) }}</time>
            </div>
            <p>{{ item.content }}</p>
            <button
              v-if="item.trace"
              class="trace-link"
              type="button"
              :aria-pressed="selectedMessageId === item.id"
              @click="selectedMessageId = item.id"
            >
              查看本轮执行轨迹
              <span aria-hidden="true">→</span>
            </button>
          </div>
        </article>

        <article v-if="loading" class="message-row assistant">
          <span class="message-avatar thinking" aria-hidden="true">知</span>
          <div class="message-content thinking-copy">
            <div class="message-meta"><strong>多 Agent 正在协作</strong></div>
            <span></span><span></span><span></span>
          </div>
        </article>
      </div>

      <form class="composer" @submit.prevent="submit">
        <label class="sr-only" for="chat-message">输入客服问题</label>
        <textarea
          id="chat-message"
          :value="draft"
          rows="2"
          maxlength="2000"
          placeholder="描述你的问题，Enter 发送，Shift + Enter 换行"
          @input="$emit('update:draft', $event.target.value)"
          @keydown.enter.exact="handleEnter"
        ></textarea>
        <button class="send-button" type="submit" :disabled="loading || !draft.trim()" aria-label="发送消息">
          <span v-if="loading" class="spinner" aria-hidden="true"></span>
          <svg v-else viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12l16-8-5.5 16-3-6.5L4 12zm7.5 1.5L20 4" /></svg>
          <span>{{ loading ? '处理中' : '发送' }}</span>
        </button>
        <div class="composer-footer">
          <span>答案由 AI 生成，请在关键业务操作前复核</span>
          <span>{{ draft.length }}/2000</span>
        </div>
      </form>

      <p v-if="error" class="inline-alert error" role="alert">{{ error }}</p>
    </section>

    <ExecutionTrace :trace="activeTrace" :loading="loading" />
  </div>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import ExecutionTrace from './ExecutionTrace.vue'
import { agentLabel, intentLabel } from '../lib/presentation'

const props = defineProps({
  messages: { type: Array, required: true },
  draft: { type: String, default: '' },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
  conversationId: { type: String, default: '' }
})

const emit = defineEmits(['send', 'clear', 'update:draft'])
const messageList = ref(null)
const selectedMessageId = ref('')

const prompts = [
  '退款多久到账？',
  '订单 #12345 支付成功但状态异常',
  '我无法登录账号，需要技术支持'
]

const tracedMessages = computed(() => props.messages.filter((item) => item.trace))
const activeTrace = computed(() => {
  const selected = tracedMessages.value.find((item) => item.id === selectedMessageId.value)
  return (selected || tracedMessages.value.at(-1))?.trace || null
})

watch(
  () => [props.messages.length, props.loading],
  async () => {
    const latest = tracedMessages.value.at(-1)
    if (latest) selectedMessageId.value = latest.id
    await nextTick()
    messageList.value?.scrollTo({ top: messageList.value.scrollHeight, behavior: 'smooth' })
  }
)

function submit() {
  const value = props.draft.trim()
  if (value && !props.loading) emit('send', value)
}

function handleEnter(event) {
  if (event.isComposing) return
  event.preventDefault()
  submit()
}

function formatTime(value) {
  if (!value) return ''
  return new Intl.DateTimeFormat('zh-CN', {
    hour: '2-digit',
    minute: '2-digit'
  }).format(new Date(value))
}
</script>
