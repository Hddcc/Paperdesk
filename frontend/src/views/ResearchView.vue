<template>
  <section class="page-shell research-workbench">
    <div class="research-column research-column-left">
      <article class="panel">
        <header class="section-head">
          <div>
            <h2>研究工作台</h2>
            <p>这里会持续展示主题、参数、任务推进与最终综述。</p>
          </div>
        </header>

        <div class="panel-body status-stack">
          <div class="status-strip">
            <strong>当前状态</strong>
            <span>{{ store.status }}</span>
          </div>

          <div v-if="store.topic" class="request-summary">
            <div class="request-summary-row">
              <span class="request-label">研究主题</span>
              <strong>{{ store.topic }}</strong>
            </div>
            <div class="request-summary-grid">
              <p>
                <span class="request-label">在线论文</span>
                {{ store.lastRequest.top_k_online || 3 }} 篇
              </p>
              <p>
                <span class="request-label">本地证据</span>
                {{ store.lastRequest.top_k_local || 3 }} 条
              </p>
              <p>
                <span class="request-label">检索来源</span>
                {{ formatSearchProvider(store.lastRequest.search_provider || null) }}
              </p>
            </div>
            <p v-if="store.lastRequest.notes" class="task-intent">
              {{ store.lastRequest.notes }}
            </p>
          </div>

          <p v-if="store.error" class="error-text">{{ store.error }}</p>
        </div>
      </article>

      <ResearchLaunchPanel
        compact
        title="补充发起入口"
        description="你也可以直接在工作台里修改参数并重新开始一轮研究。"
        submit-label="开始研究"
        :disabled="store.isRunning"
        :initial-request="store.lastRequest"
        @submit="handleSubmit"
      />

      <ResearchTaskList
        :active-task-id="store.activeTaskId"
        :completed-count="store.completedCount"
        :tasks="store.tasks"
        @select="store.setActiveTask"
      />
    </div>

    <TaskSummaryPanel :logs="store.logs" :task-entry="store.activeTask" />

    <article class="panel research-report-panel">
      <header class="section-head">
        <div>
          <h2>最终综述预览</h2>
          <p>研究完成后，可在这里直接查看当前轮次的 Markdown 综述。</p>
        </div>
      </header>
      <div class="panel-body panel-scroll">
        <MarkdownPreview v-if="store.reportMarkdown" :markdown="store.reportMarkdown" />
        <p v-else class="empty-state">综述生成后会显示在这里。</p>
      </div>
    </article>
  </section>
</template>

<script setup lang="ts">
import { onMounted } from "vue";

import MarkdownPreview from "../components/MarkdownPreview.vue";
import ResearchLaunchPanel from "../components/ResearchLaunchPanel.vue";
import ResearchTaskList from "../components/ResearchTaskList.vue";
import TaskSummaryPanel from "../components/TaskSummaryPanel.vue";
import { useResearchStore } from "../stores/research";
import type { ResearchRequest } from "../types/models";

const store = useResearchStore();

onMounted(() => {
  if (store.hasPendingRequest) {
    void store.startPendingResearch();
  }
});

async function handleSubmit(payload: ResearchRequest) {
  await store.startResearch(payload);
}

function formatSearchProvider(value: string | null) {
  switch (value) {
    case "all":
      return "全部来源";
    case "openalex":
      return "仅 OpenAlex";
    case "arxiv":
      return "仅 arXiv";
    default:
      return "自动选择";
  }
}
</script>
