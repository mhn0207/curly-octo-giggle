<template>
  <section class="feature-workspace" aria-labelledby="knowledge-title">
    <header class="feature-hero knowledge-hero">
      <div>
        <span class="section-kicker">KNOWLEDGE OPERATIONS</span>
        <h2 id="knowledge-title">让知识成为 Agent 的可靠依据</h2>
        <p>验证 RAG 召回质量，维护企业服务规范，并将新文档即时加入知识库。</p>
      </div>
      <div class="hero-stat">
        <strong>{{ knowledgeCount }}</strong>
        <span>知识片段</span>
      </div>
    </header>

    <div class="workspace-tabs" role="tablist" aria-label="知识库工具">
      <button
        type="button"
        role="tab"
        :aria-selected="activePane === 'search'"
        :class="{ active: activePane === 'search' }"
        @click="activePane = 'search'"
      >
        语义检索
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="activePane === 'import'"
        :class="{ active: activePane === 'import' }"
        @click="activePane = 'import'"
      >
        导入知识
      </button>
    </div>

    <section v-show="activePane === 'search'" class="knowledge-pane" role="tabpanel">
      <div class="pane-heading">
        <div>
          <span class="section-kicker">RAG QUALITY CHECK</span>
          <h3>语义检索测试</h3>
          <p>查看查询改写、并行召回与重排后的 Top-K 结果。</p>
        </div>
        <span v-if="reranked" class="status-badge online"><i></i>已重排</span>
      </div>

      <form class="search-form" @submit.prevent="$emit('search')">
        <label for="knowledge-query">检索问题</label>
        <div class="search-field">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10.5 18a7.5 7.5 0 110-15 7.5 7.5 0 010 15zm5.5-2l5 5" /></svg>
          <input
            id="knowledge-query"
            :value="searchQuery"
            placeholder="例如：大促期间退款多久到账？"
            @input="$emit('update:searchQuery', $event.target.value)"
          />
          <button class="button button-primary" type="submit" :disabled="searchLoading || !searchQuery.trim()">
            <span v-if="searchLoading" class="spinner" aria-hidden="true"></span>
            {{ searchLoading ? '检索中' : '开始检索' }}
          </button>
        </div>
      </form>

      <p v-if="searchError" class="inline-alert error" role="alert">{{ searchError }}</p>
      <div v-else-if="searchLoading" class="result-skeletons" aria-label="正在加载检索结果">
        <span v-for="index in 3" :key="index"></span>
      </div>
      <div v-else-if="results.length" class="search-results" aria-live="polite">
        <article v-for="(item, index) in results" :key="item.id || item.title || index" class="search-result">
          <div class="result-rank">{{ String(index + 1).padStart(2, '0') }}</div>
          <div class="result-body">
            <div class="result-heading">
              <h4>{{ item.title || '未命名知识片段' }}</h4>
              <span>相关度 {{ formatScore(item.score) }}</span>
            </div>
            <div class="score-track" aria-hidden="true"><i :style="{ width: scoreWidth(item.score) }"></i></div>
            <p>{{ item.content }}</p>
            <small v-if="item.chunk !== undefined">片段 #{{ item.chunk }}</small>
          </div>
        </article>
      </div>
      <div v-else class="pane-empty">
        <span aria-hidden="true">⌕</span>
        <strong>还没有检索结果</strong>
        <p>输入一个真实业务问题，观察知识库能否召回正确规范。</p>
      </div>
    </section>

    <section v-show="activePane === 'import'" class="knowledge-pane" role="tabpanel">
      <div class="pane-heading">
        <div>
          <span class="section-kicker">CONTENT INGESTION</span>
          <h3>导入业务知识</h3>
          <p>可以直接录入一条政策，也可以上传 TXT、Markdown 或 JSON 文档。</p>
        </div>
      </div>

      <div class="import-grid">
        <form class="document-form" @submit.prevent="$emit('submit-document')">
          <label for="document-title">知识标题</label>
          <input
            id="document-title"
            :value="docTitle"
            placeholder="例如：大促退款补充政策"
            @input="$emit('update:docTitle', $event.target.value)"
          />

          <label for="document-content">知识内容</label>
          <textarea
            id="document-content"
            :value="docContent"
            rows="8"
            placeholder="输入完整、可核验的业务政策内容"
            @input="$emit('update:docContent', $event.target.value)"
          ></textarea>

          <button
            class="button button-primary"
            type="submit"
            :disabled="importLoading || !docTitle.trim() || !docContent.trim()"
          >
            <span v-if="importLoading" class="spinner" aria-hidden="true"></span>
            {{ importLoading ? '正在导入' : '添加到知识库' }}
          </button>
        </form>

        <div class="upload-column">
          <label class="file-drop">
            <input type="file" accept=".txt,.md,.json" @change="handleFile" />
            <span class="upload-icon" aria-hidden="true">↑</span>
            <strong>{{ fileName || '选择或拖入知识文件' }}</strong>
            <span>支持 .txt / .md / .json，最大 10 MB</span>
            <small>{{ importLoading ? '正在上传并切片…' : '文件会自动解析、切片并加入向量库' }}</small>
          </label>
          <div class="format-note">
            <strong>JSON 格式示例</strong>
            <code>[{"title": "...", "content": "..."}]</code>
          </div>
        </div>
      </div>

      <p v-if="localFileError || importError" class="inline-alert error" role="alert">
        {{ localFileError || importError }}
      </p>
    </section>
  </section>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  knowledgeCount: { type: [String, Number], default: '—' },
  searchQuery: { type: String, default: '' },
  results: { type: Array, default: () => [] },
  reranked: { type: Boolean, default: false },
  searchLoading: { type: Boolean, default: false },
  searchError: { type: String, default: '' },
  docTitle: { type: String, default: '' },
  docContent: { type: String, default: '' },
  importLoading: { type: Boolean, default: false },
  importError: { type: String, default: '' }
})

const emit = defineEmits([
  'search',
  'submit-document',
  'upload',
  'update:searchQuery',
  'update:docTitle',
  'update:docContent'
])

const activePane = ref('search')
const fileName = ref('')
const localFileError = ref('')

function handleFile(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  if (file.size > 10 * 1024 * 1024) {
    localFileError.value = '文件超过 10 MB 限制，请压缩或拆分后重试。'
    fileName.value = ''
    return
  }
  localFileError.value = ''
  fileName.value = file.name
  emit('upload', file)
}

function formatScore(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number.toFixed(3) : '—'
}

function scoreWidth(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '0%'
  return String(Math.max(0, Math.min(100, number * 100))) + '%'
}
</script>
