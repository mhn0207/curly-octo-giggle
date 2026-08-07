<template>
  <aside id="app-sidebar" :class="['sidebar', { open }]" aria-label="工作台导航">
    <div class="sidebar-heading">
      <a class="brand" href="#" aria-label="知应 AI 首页" @click.prevent="$emit('navigate', 'chat')">
        <span class="brand-mark" aria-hidden="true">知</span>
        <span>
          <strong>知应 AI</strong>
          <small>多 Agent 智能客服</small>
        </span>
      </a>
      <button class="icon-button sidebar-close" type="button" aria-label="关闭导航" @click="$emit('close')">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M6 6l12 12M18 6L6 18" />
        </svg>
      </button>
    </div>

    <nav class="primary-nav" aria-label="主要功能">
      <button
        v-for="item in navItems"
        :key="item.id"
        type="button"
        :class="{ active: activeView === item.id }"
        :aria-current="activeView === item.id ? 'page' : undefined"
        @click="$emit('navigate', item.id)"
      >
        <span class="nav-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24"><path :d="item.path" /></svg>
        </span>
        <span>
          <strong>{{ item.label }}</strong>
          <small>{{ item.caption }}</small>
        </span>
      </button>
    </nav>

    <section class="connection-card" aria-labelledby="connection-title">
      <div class="connection-title-row">
        <div>
          <span class="section-kicker">SYSTEM</span>
          <h2 id="connection-title">服务连接</h2>
        </div>
        <span :class="['status-badge', healthState]">
          <i aria-hidden="true"></i>{{ healthLabel }}
        </span>
      </div>

      <div class="mini-metrics">
        <div>
          <span>知识片段</span>
          <strong>{{ knowledgeCount }}</strong>
        </div>
        <div>
          <span>已加载 Agent</span>
          <strong>{{ agentCount }}</strong>
        </div>
      </div>

      <details class="settings-details">
        <summary>连接与会话设置</summary>
        <div class="settings-body">
          <label for="api-endpoint">
            <span>API 地址</span>
            <input
              id="api-endpoint"
              :value="settings.endpoints.zhiying"
              placeholder="/api/zhiying"
              @change="$emit('update-setting', 'endpoint', $event.target.value)"
            />
          </label>
          <label for="user-id">
            <span>用户 ID</span>
            <input
              id="user-id"
              :value="settings.userId"
              placeholder="u1001"
              @change="$emit('update-setting', 'userId', $event.target.value)"
            />
          </label>
          <label for="conversation-id">
            <span>会话 ID</span>
            <input
              id="conversation-id"
              :value="settings.conversationId"
              placeholder="发送后自动生成"
              @change="$emit('update-setting', 'conversationId', $event.target.value)"
            />
          </label>
        </div>
      </details>

      <div class="sidebar-actions">
        <button class="button button-primary button-block" type="button" :disabled="loading" @click="$emit('refresh')">
          <span v-if="loading" class="spinner" aria-hidden="true"></span>
          {{ loading ? '正在同步' : '刷新运行状态' }}
        </button>
        <a class="button button-ghost button-block" :href="docsUrl" target="_blank" rel="noreferrer">打开 API 文档</a>
      </div>

      <p v-if="error" class="connection-error" role="status">{{ error }}</p>
    </section>

    <footer class="sidebar-footer">
      <span>Workspace v2.0</span>
      <span>{{ backendLabel }}</span>
    </footer>
  </aside>
</template>

<script setup>
defineProps({
  activeView: { type: String, required: true },
  open: { type: Boolean, default: false },
  settings: { type: Object, required: true },
  healthState: { type: String, default: 'checking' },
  healthLabel: { type: String, default: '检查中' },
  knowledgeCount: { type: [String, Number], default: '—' },
  agentCount: { type: [String, Number], default: '—' },
  backendLabel: { type: String, default: '知应 AI' },
  docsUrl: { type: String, required: true },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' }
})

defineEmits(['navigate', 'close', 'refresh', 'update-setting'])

const navItems = [
  {
    id: 'chat',
    label: '对话工作台',
    caption: '路由、工具与综合过程',
    path: 'M5 5.75A2.75 2.75 0 017.75 3h8.5A2.75 2.75 0 0119 5.75v6.5A2.75 2.75 0 0116.25 15H10l-4.5 3v-3.5A2.75 2.75 0 013 12.25v-6.5A2.75 2.75 0 015.75 3'
  },
  {
    id: 'knowledge',
    label: '知识运营',
    caption: 'RAG 检索与文档导入',
    path: 'M4 5.5A2.5 2.5 0 016.5 3H11v16H6.5A2.5 2.5 0 014 16.5v-11zm16 0A2.5 2.5 0 0017.5 3H13v16h4.5a2.5 2.5 0 002.5-2.5v-11z'
  },
  {
    id: 'monitor',
    label: '运行监控',
    caption: 'Agent、工具与告警',
    path: 'M4 18V9m5 9V5m5 13v-6m5 6V3'
  }
]
</script>
