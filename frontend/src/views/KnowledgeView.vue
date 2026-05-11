<template>
  <section class="page-shell knowledge-workbench">
    <article class="panel">
      <header class="section-head">
        <div>
          <h2>自由问答</h2>
          <p>基于本地知识库检索证据，再返回中文答案与页码引用。</p>
        </div>
      </header>

      <div class="panel-body panel-scroll knowledge-panel-body">
        <label>
          <span class="request-label">问题</span>
          <textarea v-model="ragQuestion" placeholder="例如：这些论文如何讨论 RAG 评估方法？" />
        </label>
        <label>
          <span class="request-label">补充说明（可选）</span>
          <textarea v-model="ragNotes" placeholder="例如：优先关注评估指标与证据归因。" />
        </label>
        <button class="button-primary" :disabled="store.asking || !ragQuestion.trim()" @click="handleAsk">
          {{ store.asking ? "检索中..." : "开始问答" }}
        </button>

        <div v-if="store.ragAnswer" class="summary-markdown">
          <h3>回答</h3>
          <p>{{ store.ragAnswer.answer }}</p>
          <p class="card-meta">
            检索到 {{ store.ragAnswer.retrieval_count }} 条证据 · 来源：
            {{ store.ragAnswer.sources.join("、") || "暂无" }}
          </p>
          <ul class="evidence-list">
            <li v-for="item in store.ragAnswer.evidence_items" :key="item.id" class="evidence-card">
              <strong>{{ item.citation_label }}</strong>
              <p class="card-meta">{{ item.title }}</p>
              <p>{{ item.quote }}</p>
            </li>
          </ul>
        </div>
      </div>
    </article>

    <article class="panel">
      <header class="section-head">
        <div>
          <h2>本地论文分析</h2>
          <p>选择 1 篇做结构化分析，或选择多篇做比较。</p>
        </div>
      </header>

      <div class="panel-body panel-scroll knowledge-panel-body">
        <div class="document-pick-grid">
          <label
            v-for="document in readyDocuments"
            :key="document.id"
            class="pick-card"
          >
            <input v-model="selectedDocumentIds" type="checkbox" :value="document.id" />
            <div>
              <strong>{{ document.display_name }}</strong>
              <p class="card-meta">{{ document.title || "暂未提取标题" }}</p>
              <p class="card-meta">
                {{ document.page_count || 0 }} 页 · 索引 {{ formatParserStatus(document.parser_status) }}
              </p>
            </div>
          </label>
        </div>

        <label>
          <span class="request-label">分析问题（可选）</span>
          <textarea
            v-model="analysisQuestion"
            placeholder="例如：请比较这些论文的方法差异、实验设置和局限。"
          />
        </label>

        <div class="hero-actions">
          <button
            class="button-secondary"
            :disabled="store.analyzing || selectedDocumentIds.length !== 1"
            @click="handleAnalyze('single')"
          >
            单篇分析
          </button>
          <button
            class="button-primary"
            :disabled="store.analyzing || selectedDocumentIds.length < 2"
            @click="handleAnalyze('compare')"
          >
            多篇比较
          </button>
        </div>

        <div v-if="store.analysisResult" class="summary-markdown">
          <h3>{{ store.analysisResult.mode === "compare" ? "比较结果" : "分析结果" }}</h3>
          <p>{{ store.analysisResult.answer }}</p>
          <ul class="bullet-list">
            <li v-for="section in store.analysisResult.sections" :key="section.title">
              <strong>{{ section.title }}：</strong>{{ section.content }}
            </li>
          </ul>
        </div>
      </div>
    </article>

    <article class="panel">
      <header class="section-head">
        <div>
          <h2>在线候选筛选</h2>
          <p>先拉取在线论文候选，再给出是否值得加入本地知识库的建议。</p>
        </div>
      </header>

      <div class="panel-body panel-scroll knowledge-panel-body">
        <label>
          <span class="request-label">主题</span>
          <textarea v-model="curationTopic" placeholder="例如：RAG 评估基准" />
        </label>
        <div class="request-grid">
          <label>
            <span class="request-label">检索源</span>
            <select v-model="curationProvider">
              <option value="all">all</option>
              <option value="auto">auto</option>
              <option value="openalex">openalex</option>
              <option value="arxiv">arxiv</option>
            </select>
          </label>
          <label>
            <span class="request-label">候选数</span>
            <input v-model.number="curationTopK" type="number" min="1" max="10" />
          </label>
        </div>
        <button class="button-primary" :disabled="store.curating || !curationTopic.trim()" @click="handleCurate">
          {{ store.curating ? "筛选中..." : "筛选候选" }}
        </button>

        <ul v-if="store.curationResult?.items.length" class="paper-list">
          <li v-for="item in store.curationResult.items" :key="item.paper.title" class="paper-card">
            <div class="task-title-row">
              <strong class="card-title">{{ item.paper.title }}</strong>
              <span class="status-badge" :data-status="item.decision">
                {{ formatDecision(item.decision) }}
              </span>
            </div>
            <p class="card-meta">{{ (item.paper.authors || []).join("、") || "作者未知" }}</p>
            <p class="card-meta">{{ item.reason }}</p>
          </li>
        </ul>

        <p v-if="store.error" class="error-text">{{ store.error }}</p>
      </div>
    </article>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";

import { useDocumentStore } from "../stores/documents";
import { useKnowledgeStore } from "../stores/knowledge";

const documentStore = useDocumentStore();
const store = useKnowledgeStore();

const ragQuestion = ref("");
const ragNotes = ref("");
const analysisQuestion = ref("");
const selectedDocumentIds = ref<string[]>([]);
const curationTopic = ref("");
const curationProvider = ref("all");
const curationTopK = ref(5);

const readyDocuments = computed(() =>
  documentStore.documents.filter((document) => document.status === "ready")
);

onMounted(async () => {
  await documentStore.refreshDocuments();
});

async function handleAsk() {
  await store.ask({
    question: ragQuestion.value,
    notes: ragNotes.value,
    document_ids: selectedDocumentIds.value,
    top_k: 4
  });
}

async function handleAnalyze(mode: "single" | "compare") {
  await store.analyze({
    document_ids: selectedDocumentIds.value,
    mode,
    question: analysisQuestion.value
  });
}

async function handleCurate() {
  await store.curate({
    topic: curationTopic.value,
    search_provider: curationProvider.value,
    top_k_online: curationTopK.value
  });
}

function formatParserStatus(value?: string | null) {
  switch (value) {
    case "indexed":
      return "完成";
    case "parsed":
      return "已解析";
    case "processing":
      return "处理中";
    case "failed":
      return "失败";
    default:
      return value || "待处理";
  }
}

function formatDecision(value: string) {
  switch (value) {
    case "recommended":
      return "推荐";
    case "consider":
      return "可考虑";
    case "skip":
      return "跳过";
    default:
      return value;
  }
}
</script>
