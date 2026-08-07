<template>
  <main class="app-shell">
    <button
      v-if="sidebarOpen"
      class="sidebar-backdrop"
      type="button"
      aria-label="关闭导航"
      @click="sidebarOpen = false"
    ></button>

    <AppSidebar
      :active-view="activeView"
      :inert="isMobile && !sidebarOpen"
      :aria-hidden="isMobile && !sidebarOpen ? 'true' : undefined"
      :open="sidebarOpen"
      :settings="settings"
      :health-state="health.state"
      :health-label="health.label"
      :knowledge-count="knowledgeCount"
      :agent-count="agentCount"
      :backend-label="currentBackend.label"
      :docs-url="docsUrl"
      :loading="operations.system"
      :error="errors.system"
      @navigate="navigate"
      @close="sidebarOpen = false"
      @refresh="refreshSystem"
      @update-setting="updateSetting"
    />

    <section class="app-main">
      <header class="topbar">
        <button
          class="icon-button mobile-menu"
          type="button"
          aria-label="打开导航"
          aria-controls="app-sidebar"
          :aria-expanded="sidebarOpen"
          @click="sidebarOpen = true"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" /></svg>
        </button>
        <div class="topbar-title">
          <span class="section-kicker">{{ viewMeta.kicker }}</span>
          <h1>{{ viewMeta.title }}</h1>
        </div>
        <div class="topbar-actions">
          <span class="endpoint-label">{{ currentBackend.baseUrl }}</span>
          <span :class="['topbar-health', health.state]">
            <i aria-hidden="true"></i>{{ health.label }}
          </span>
          <a class="icon-button docs-button" :href="docsUrl" target="_blank" rel="noreferrer" aria-label="打开 API 文档">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6m0-6l-9 9M10 5H5v14h14v-5" /></svg>
          </a>
        </div>
      </header>

      <div class="content-canvas">
        <ChatWorkspace
          v-if="activeView === 'chat'"
          :messages="messages"
          :draft="draft"
          :loading="operations.chat"
          :error="errors.chat"
          :conversation-id="settings.conversationId"
          @send="sendMessage"
          @clear="clearConversation"
          @update:draft="draft = $event"
        />

        <KnowledgeWorkspace
          v-else-if="activeView === 'knowledge'"
          :knowledge-count="knowledgeCount"
          :search-query="searchQuery"
          :results="searchResults"
          :reranked="searchReranked"
          :search-loading="operations.search"
          :search-error="errors.search"
          :doc-title="docTitle"
          :doc-content="docContent"
          :import-loading="operations.import"
          :import-error="errors.import"
          @search="searchKnowledge"
          @submit-document="submitKnowledge"
          @upload="handleUpload"
          @update:search-query="searchQuery = $event"
          @update:doc-title="docTitle = $event"
          @update:doc-content="docContent = $event"
        />

        <MonitorWorkspace
          v-else
          :monitor="monitorSummary"
          :knowledge-count="knowledgeCount"
          :tool-catalog="toolCatalog"
          :recent-executions="toolExecutions"
          :skills-summary="skillsSummary"
          :loading="operations.system"
          :error="errors.system"
          @refresh="refreshSystem"
        />
      </div>
    </section>

    <div class="toast-region" aria-live="polite" aria-atomic="true">
      <div v-for="notice in notices" :key="notice.id" :class="['toast', notice.tone]">
        <span aria-hidden="true">{{ notice.tone === 'success' ? '✓' : '!' }}</span>
        <p>{{ notice.message }}</p>
        <button type="button" aria-label="关闭提示" @click="removeNotice(notice.id)">×</button>
      </div>
    </div>
  </main>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import AppSidebar from './components/AppSidebar.vue'
import ChatWorkspace from './components/ChatWorkspace.vue'
import KnowledgeWorkspace from './components/KnowledgeWorkspace.vue'
import MonitorWorkspace from './components/MonitorWorkspace.vue'
import {
  addKnowledge,
  backendMeta,
  createInitialSettings,
  requestChat,
  requestHealth,
  requestKnowledgeStats,
  requestMonitor,
  requestSearch,
  requestSkillsSummary,
  requestToolExecutions,
  requestToolsCatalog,
  saveSettings,
  uploadKnowledge
} from './lib/backends'
import { friendlyError } from './lib/presentation'

const VIEW_META = {
  chat: { kicker: 'MULTI-AGENT SERVICE', title: '对话工作台' },
  knowledge: { kicker: 'RAG KNOWLEDGE BASE', title: '知识运营' },
  monitor: { kicker: 'SYSTEM OBSERVABILITY', title: '运行监控' }
}

const settings = reactive(createInitialSettings())
const activeView = ref('chat')
const sidebarOpen = ref(false)
const isMobile = ref(false)
const messages = ref(loadMessages())
const draft = ref('')
const searchQuery = ref('退款多久能到账')
const searchResults = ref([])
const searchReranked = ref(false)
const docTitle = ref('大促退款补充政策')
const docContent = ref('大促期间退款审核时间可能延长到 3-5 个工作日。')
const knowledgeCount = ref('—')
const monitorSummary = ref(null)
const toolCatalog = ref([])
const toolExecutions = ref([])
const skillsSummary = ref(null)
const notices = ref([])

const health = reactive({
  state: 'checking',
  label: '检查中',
  lastChecked: ''
})

const operations = reactive({
  chat: false,
  search: false,
  import: false,
  system: false
})

const errors = reactive({
  chat: '',
  search: '',
  import: '',
  system: ''
})

const currentBackend = computed(() => backendMeta(settings))
const docsUrl = computed(() => currentBackend.value.baseUrl + '/docs')
const viewMeta = computed(() => VIEW_META[activeView.value])
const agentCount = computed(() => {
  const count = Object.keys(monitorSummary.value?.agent_stats || {}).length
  return count || '—'
})

let mobileMediaQuery

onMounted(() => {
  mobileMediaQuery = globalThis.matchMedia('(max-width: 900px)')
  syncMobileState(mobileMediaQuery)
  mobileMediaQuery.addEventListener?.('change', syncMobileState)
  refreshSystem()
})

onBeforeUnmount(() => mobileMediaQuery?.removeEventListener?.('change', syncMobileState))

function syncMobileState(event) {
  isMobile.value = event.matches
  if (!event.matches) sidebarOpen.value = false
}

function createId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()
  return Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10)
}

function navigate(view) {
  activeView.value = view
  sidebarOpen.value = false
}

function updateSetting(field, value) {
  if (field === 'endpoint') settings.endpoints.zhiying = value.trim() || '/api/zhiying'
  if (field === 'userId') settings.userId = value.trim() || 'anonymous'
  if (field === 'conversationId') settings.conversationId = value.trim()
  persistSettings()
  if (field === 'endpoint') refreshSystem()
}

function persistSettings() {
  try {
    saveSettings(settings)
  } catch {
    showNotice('浏览器无法保存本地设置，本次配置仅临时生效。', 'warning')
  }
}

async function refreshSystem() {
  if (operations.system) return
  operations.system = true
  errors.system = ''
  health.state = 'checking'
  health.label = '检查中'

  const results = await Promise.allSettled([
    requestHealth(settings),
    requestKnowledgeStats(settings),
    requestMonitor(settings),
    requestToolsCatalog(settings),
    requestToolExecutions(settings, 20),
    requestSkillsSummary(settings)
  ])
  const [healthResult, statsResult, monitorResult, catalogResult, executionsResult, skillsResult] = results

  if (healthResult.status === 'fulfilled') {
    const online = healthResult.value?.status === 'ok'
    health.state = online ? 'online' : 'offline'
    health.label = online ? '运行正常' : '状态异常'
  } else {
    health.state = 'offline'
    health.label = '连接失败'
    errors.system = friendlyError(healthResult.reason)
  }

  if (statsResult.status === 'fulfilled') {
    knowledgeCount.value = statsResult.value?.total_chunks ?? statsResult.value?.totalChunks ?? '—'
  }
  if (monitorResult.status === 'fulfilled') monitorSummary.value = monitorResult.value
  if (catalogResult.status === 'fulfilled') toolCatalog.value = normalizeCatalog(catalogResult.value)
  if (executionsResult.status === 'fulfilled') {
    toolExecutions.value = executionsResult.value?.executions || []
  }
  if (skillsResult.status === 'fulfilled') skillsSummary.value = skillsResult.value

  health.lastChecked = new Date().toISOString()
  operations.system = false
}

async function sendMessage(content) {
  const value = String(content || '').trim().slice(0, 2000)
  if (!value || operations.chat) return

  messages.value.push({
    id: createId(),
    role: 'user',
    content: value,
    createdAt: new Date().toISOString()
  })
  draft.value = ''
  errors.chat = ''
  operations.chat = true
  persistMessages()

  try {
    const response = await requestChat(settings, value)
    if (response.conversationId && response.conversationId !== settings.conversationId) {
      settings.conversationId = response.conversationId
      persistSettings()
    }
    const { raw, ...trace } = response
    messages.value.push({
      id: createId(),
      role: 'assistant',
      content: response.response || '服务已完成处理，但没有返回可展示的文本。',
      trace,
      createdAt: new Date().toISOString()
    })
  } catch (error) {
    const message = friendlyError(error)
    errors.chat = message
    messages.value.push({
      id: createId(),
      role: 'assistant',
      content: message,
      failed: true,
      createdAt: new Date().toISOString()
    })
  } finally {
    operations.chat = false
    persistMessages()
  }
}

function clearConversation() {
  messages.value = []
  draft.value = ''
  errors.chat = ''
  settings.conversationId = ''
  persistSettings()
  persistMessages()
  showNotice('已开始一段新会话。')
}

async function searchKnowledge() {
  const query = searchQuery.value.trim()
  if (!query || operations.search) return
  operations.search = true
  errors.search = ''
  try {
    const data = await requestSearch(settings, query, 5)
    searchResults.value = data?.results || []
    searchReranked.value = Boolean(data?.reranked)
    if (!searchResults.value.length) showNotice('没有找到相关知识片段。', 'warning')
  } catch (error) {
    errors.search = friendlyError(error)
    searchResults.value = []
    searchReranked.value = false
  } finally {
    operations.search = false
  }
}

async function submitKnowledge() {
  if (operations.import) return
  const title = docTitle.value.trim()
  const content = docContent.value.trim()
  if (!title || !content) return
  operations.import = true
  errors.import = ''
  try {
    const data = await addKnowledge(settings, [{ title, content }])
    knowledgeCount.value = data?.total_chunks ?? knowledgeCount.value
    showNotice(data?.message || '知识已成功加入知识库。')
    await refreshSystem()
  } catch (error) {
    errors.import = friendlyError(error)
  } finally {
    operations.import = false
  }
}

async function handleUpload(file) {
  if (!file || operations.import) return
  operations.import = true
  errors.import = ''
  try {
    const data = await uploadKnowledge(settings, file)
    knowledgeCount.value = data?.total_chunks ?? knowledgeCount.value
    showNotice(data?.message || '文件已成功导入知识库。')
    await refreshSystem()
  } catch (error) {
    errors.import = friendlyError(error)
  } finally {
    operations.import = false
  }
}

function normalizeCatalog(value) {
  const tools = value?.tools ?? value
  if (Array.isArray(tools)) return tools
  if (tools && typeof tools === 'object') {
    return Object.entries(tools).map(([name, metadata]) => ({ name, ...metadata }))
  }
  return []
}

function showNotice(message, tone = 'success') {
  const notice = { id: createId(), message, tone }
  notices.value.push(notice)
  globalThis.setTimeout(() => removeNotice(notice.id), 4200)
}

function removeNotice(id) {
  notices.value = notices.value.filter((notice) => notice.id !== id)
}

function loadMessages() {
  try {
    const value = JSON.parse(localStorage.getItem('zhiying.frontend.messages') || '[]')
    return Array.isArray(value) ? value.slice(-40) : []
  } catch {
    return []
  }
}

function persistMessages() {
  try {
    localStorage.setItem('zhiying.frontend.messages', JSON.stringify(messages.value.slice(-40)))
  } catch {
    // 对话仍保留在当前页面，不阻塞主要流程。
  }
}
</script>
